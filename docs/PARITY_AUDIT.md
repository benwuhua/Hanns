# PARITY_AUDIT (Non-GPU)

Last updated: 2026-03-09 12:02
Sync baseline: 4f60908fc9ad7438b4b8ff64210481ab281009b0 from origin/main

## 轮次记录
- 2026-03-09 12:02: **计划轮次：关闭 `ABI-P3-002` 并切换 `PERSIST-P3-003`（builder-plan）**
  1. 复核输入：`memory/PLAN_RESULT.json`、`memory/EXEC_RESULT.json`、`memory/DEV_RESULT.json`、`memory/VERIFY_RESULT.json`、`TASK_QUEUE.md`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md`、`docs/PARITY_AUDIT.md`、`src/ffi.rs`、`src/index.rs`。
  2. 调度判断：最新 `EXEC_RESULT.updated_at=2026-03-09T03:21:00Z` 晚于 `PLAN_RESULT.updated_at=2026-03-09T02:46:00Z`，因此本轮必须重新 planning，不能 skip。
  3. 收口结论：`src/ffi.rs` 已形成逐索引 `additional_scalar` / `capabilities` / `semantics` contract，且 `ffi::tests::test_ffi_abi_metadata_contract` 已覆盖 null-safe、unsupported、partial-supported 与 per-index 差异场景；`ABI-P3-002` 当前验收已达成。
  4. 阶段决策：按 Phase 5 的生产硬化目标，下一最小高价值缺口切换为 `PERSIST-P3-003`，先收敛 `file_save_load` / `memory_serialize` / `deserialize_from_file` 的支持矩阵与 focused regressions，而不是继续在已完成 ABI 范围空转。
  5. 治理动作：同步更新 `TASK_QUEUE.md`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md` 与 `memory/PLAN_RESULT.json`，将当前任务切换为更窄的 persistence 子切片。
  状态：Phase 5 Active（ABI metadata hardening closed；persistence semantics hardening promoted）。
- 2026-03-09 02:46: **计划轮次：ABI-P3-002 重新收口，清除陈旧 blocker 叙事（builder-plan）**
  1. 复核输入：`memory/PLAN_RESULT.json`、`memory/EXEC_RESULT.json`、`memory/DEV_RESULT.json`、`memory/VERIFY_RESULT.json`、`TASK_QUEUE.md`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md`、`docs/PARITY_AUDIT.md`、`src/ffi.rs`。
  2. 调度判断：最新 `EXEC_RESULT.updated_at=2026-03-09T02:42:05Z` 晚于 `PLAN_RESULT.updated_at=2026-03-09T02:34:00Z`，且 exec 摘要声称 required lib gate 被 HNSW parallel compatibility failure 阻塞，因此本轮不能 skip，必须先做直接调度自检与现状复核。
  3. 现状复核：计划侧实测 `cargo test --lib -q` 在当前工作树通过（524 passed, 0 failed, 2 ignored），未复现“required lib gate 被 HNSW parallel 失败阻塞”的结论；说明 blocker 已消失或结果文件已陈旧，当前真正缺口仍是 ABI 语义矩阵未收口。
  4. 阶段决策：不回切到新的 BUG，而是把 `ABI-P3-002` 拆成更窄的 per-index ABI 子矩阵（HNSW / IVF / ScaNN / Sparse 的 additional-scalar + index_meta 字段语义 + focused FFI regressions），避免 exec 围绕陈旧 blocker 空转。
  5. 治理动作：同步更新 `TASK_QUEUE.md`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md` 与 `memory/PLAN_RESULT.json`，将当前任务重写为可执行的 ABI 子切片，并把 required checks 重新绑定到现状实测通过的 gate。
  状态：Phase 5 Active（ABI metadata hardening 继续；stale gate blocker cleared at planning layer）。
- 2026-03-09 02:34: **计划轮次：关闭 `SEM-P3-001` 并切换 `ABI-P3-002`（builder-plan）**
  1. 复核输入：`memory/PLAN_RESULT.json`、`memory/EXEC_RESULT.json`、`memory/DEV_RESULT.json`、`memory/VERIFY_RESULT.json`、`TASK_QUEUE.md`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md`、`docs/PARITY_AUDIT.md`。
  2. 调度判断：最新 `EXEC_RESULT.updated_at=2026-03-09T02:14:52Z` 晚于 `PLAN_RESULT.updated_at=2026-03-09T02:02:00Z`，且 queue 首个 TODO 仍指向旧 umbrella `SEM-P3-001`，因此当前 plan 已失效，不能 skip。
  3. 收口结论：最新 exec 已把 `HNSW` / `IVF` / `Sparse` / `ScaNN` 的 `GetVectorByIds` / `HasRawData` focused semantic tail 收敛到可审计状态；继续保留 `SEM-P3-001` 作为当前任务只会让 exec 重复进入已关闭范围。
  4. 阶段决策：按 `BUG > PARITY > OPT > BENCH` 与 Phase 5 目标，下一最小高价值缺口切换为 `ABI-P3-002`，先把 FFI metadata / additional-scalar 从“最小稳定摘要”提升为逐模块真实 contract。
  5. 治理动作：同步更新 `TASK_QUEUE.md`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md`，将 `SEM-P3-001` 从 active queue 出队并把 `ABI-P3-002` 提升为当前任务。
  状态：Phase 5 Active（semantic tail closed；production metadata hardening in progress）。
- 2026-03-09 01:50: **SEM-P3-001 focused delta：收敛 DiskANN / AISAQ 的 `HasRawData` 度量语义（builder-exec）**
  1. 对照原生 `src/index/diskann/diskann.cc` 与 `src/index/diskann/diskann_aisaq.cc`，确认 `HasRawData` 只对 `L2/COSINE` 返回 true，而非“是否已有内存向量”这种运行态条件。
  2. Rust 侧修复：`src/faiss/diskann.rs` 与 `src/faiss/aisaq.rs` 改为按 metric gate 暴露 raw-data 语义；`get_vector_by_ids` 在 `has_raw_data=false` 时返回 Unsupported，而不是继续走伪 raw-data 路径。
  3. 新增 focused conformance tests：覆盖 `L2/COSINE/IP` 三种 metric 的 `has_raw_data` 返回，并锁定 DiskANN 缺失 ID 走 error 而非静默零填充。
  4. 当前剩余 delta：HNSW/IVF/Sparse/ScaNN 的 missing-id / lossy-index / empty-index 语义仍需继续矩阵化，`SEM-P3-001` 暂不关闭。
  状态：Phase 5 Active（semantic fidelity audit in progress）。
- 2026-03-08 20:36: **OPT-P2-004 收口：legality/governance 漂移已消除（builder-exec）**
  1. 复核入口：`src/api/index.rs` 中 `IndexConfig::validate()` 仍统一调用 legality matrix；`src/ffi.rs` 中 `IndexWrapper::new()` 仍在构造期调用 `validate_index_config`，未发现新运行时代码缺口。
  2. 治理动作：`TASK_QUEUE.md` 将 `OPT-P2-004` 标记完成并清空 TODO；`DEV_ROADMAP.md` 与 `GAP_ANALYSIS.md` 同步关闭最后一条 governance tail。
  3. 审计收敛：模块表 `Index factory/legality` 保持 `Done`，并在 changelog 中明确旧 `Partial` 属历史残留而非未完成开发项。
  4. 结果：queue/roadmap/gap/audit 对 legality 状态重新一致，Phase 4 可关闭。
  状态：Phase 4 Closed（no active non-GPU parity/governance tail）。
- 2026-03-08 19:41: **计划轮次：P1/P2 主线收口后重建尾部 Partial 清理队列（builder-plan）**
  1. 复核输入：`memory/PLAN_RESULT.json`、`memory/EXEC_RESULT.json`、`memory/DEV_RESULT.json`、`memory/VERIFY_RESULT.json`、`TASK_QUEUE.md`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md`、`docs/FFI_CAPABILITY_MATRIX.md`、`src/faiss/hnsw_pq.rs`。
  2. 调度判断：最新 `EXEC_RESULT.updated_at` 晚于 `PLAN_RESULT.updated_at`，且当前 queue 已无未完成 TODO，不满足 skip 条件。
  3. 收口结论：`PARITY-P1-011` 已由 exec 完成，P0/P1 主线与 Phase 3 benchmark/gate 能力均已闭环，继续维持空队列会让审计表中的尾部 `Partial` 脱离治理。
  4. 差距重估：当前最小且真实的剩余缺口是 `HNSW-PQ=Partial`（高级接口/持久化语义未闭环）；`Index factory/legality=Partial` 更像历史残留状态漂移，应作为治理清理项而非重新回切 P1。
  5. 治理动作：在 `TASK_QUEUE.md` 新增 `PARITY-P2-001` 与 `OPT-P2-004`，`DEV_ROADMAP.md` 打开新的尾部收口阶段，`GAP_ANALYSIS.md` 改写为 tail-closure 叙事。
  状态：Phase 4 Active（tail partial cleanup / quantized parity polish）。
- 2026-03-08 18:49: **计划轮次：BENCH-P2-003 收口后回切 P1 parity 缺口（builder-plan）**
  1. 复核输入：`memory/PLAN_RESULT.json`、`memory/EXEC_RESULT.json`、`memory/DEV_RESULT.json`、`memory/VERIFY_RESULT.json`、`TASK_QUEUE.md`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md`。
  2. 调度判断：最新 `EXEC_RESULT.updated_at` / `VERIFY_RESULT.updated_at` 均晚于 `PLAN_RESULT.updated_at`，触发重新 planning；不满足 skip 条件。
  3. 收口结论：`BENCH-P2-003` 的 cross-dataset artifact 已落地且 verify 通过，P2 三个 scoped tasks（`OPT-P2-003` / `BENCH-P2-002` / `BENCH-P2-003`）均可判定为完成。
  4. 差距重估：模块状态仍存在 `Sparse=Partial`、`FFI ABI=Partial`，与“Phase2 已闭环”叙事不一致；按 `BUG > PARITY > OPT > BENCH` 应优先回切 parity。
  5. 治理动作：在 `TASK_QUEUE.md` 新增并提升 `PARITY-P1-010` / `PARITY-P1-011`，同步更新 `DEV_ROADMAP.md`（Phase2 Reopened）与 `GAP_ANALYSIS.md`（主缺口从 P2 切回 P1）。
  状态：Phase 2 Reopened（parity debt burn-down）；Phase 3 Closed（validation/perf hardening 已收口）。
- 2026-03-08 17:35: **计划轮次：BENCH-P2-002 验收关闭并切换 BENCH-P2-003（builder-plan）**
  1. 复核输入：`memory/PLAN_RESULT.json`、`memory/DEV_RESULT.json`、`memory/VERIFY_RESULT.json`、`TASK_QUEUE.md`、`tests/bench_recall_gated_baseline.rs`、`benchmark_results/recall_gated_baseline.json`、`src/benchmark/report_schema.rs`。
  2. 调度判断：最新 `VERIFY_RESULT.updated_at` 晚于 `PLAN_RESULT.updated_at`，触发重新 planning；不满足 skip 条件。
  3. 收口结论：`BENCH-P2-002` 验收达成（ScaNN/RaBitQ/Sparse 条目已入 baseline，低可信条目含可执行解释），应从当前任务出队。
  4. 阶段决策：Phase 3 仍 active，下一最小高价值任务切换为 `BENCH-P2-003`（cross-dataset artifact 流水线）。
  5. 治理动作：同步更新 `TASK_QUEUE.md`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md`，保持 queue/roadmap/gap/audit 一致。
  状态：Phase 3 Active（Validation/Performance Hardening，进入 cross-dataset 收敛）。
- 2026-03-08 17:00: **任务收口守护：OPT-P2-003 验收完成并切换 BENCH-P2-002（progress-guard）**
  1. 复核输入：`TASK_QUEUE.md`、`memory/PLAN_RESULT.json`、`memory/DEV_RESULT.json`、`memory/VERIFY_RESULT.json`、`scripts/gate_profile_runner.sh`、`scripts/README_GATE_PROFILES.md`。
  2. 结论：不存在空转；本轮属于任务收口后的阶段内连续推进，`OPT-P2-003` 已满足验收。
  3. 差距重估：Phase 3 当前主缺口转为 benchmark 覆盖与可信度解释（ScaNN/RaBitQ/Sparse），而非门禁执行器。
  4. 动作：`TASK_QUEUE.md` 将 `OPT-P2-003` 标记 Done，当前任务切换为 `BENCH-P2-002`；`memory/PLAN_RESULT.json` 同步更新 next executor 为 dev。
  5. 治理同步：更新 `DEV_ROADMAP.md`（活跃任务列表）与 `GAP_ANALYSIS.md`（门禁缺口关闭），保持 queue/roadmap/gap/audit 一致。
  状态：Phase 3 Active（Validation/Performance Hardening，benchmark 扩面中）。
- 2026-03-08 16:40: **阶段切换守护：P0/P1 收口后重建 P2 队列（progress-guard）**
  1. 复核输入：`TASK_QUEUE.md`、`memory/PLAN_RESULT.json`、`memory/DEV_RESULT.json`、`memory/VERIFY_RESULT.json`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md`。
  2. 结论：最近 dev/verify 存在真实推进（`OPT-P2-002` 新增稳定回归矩阵并通过 required checks），不构成“空转”。
  3. 新差距：队列清空后若停在 `NONE` 会造成阶段治理脱节；当前主要缺口已从 parity 修复转为 validation/perf hardening。
  4. 动作：新增并排定 P2 scoped tasks（`OPT-P2-003` / `BENCH-P2-002` / `BENCH-P2-003`），并将当前任务切换为 `OPT-P2-003`。
  5. 审计约束：后续轮次需在报告中同时引用 gate profile 与 benchmark artifact，避免“门禁结论可过但性能结论不可复现”。
  状态：Phase 3 Active（Validation/Performance Hardening）。
- 2026-03-07 15:19: **BUG-P1-001 长跑卡住根因收敛（慢测治理第二轮）**
  1. 同步 baseline：`git fetch origin main` -> `4f60908fc9ad7438b4b8ff64210481ab281009b0`
  2. 本轮修复：
     - `tests/bench_hnsw_parallel.rs` 将 `test_hnsw_parallel_build_small/medium/large` 与 `test_hnsw_thread_scaling` 标记为 `#[ignore]`
     - `tests/perf_test.rs` 将 `test_performance_comparison_small/100k/1m`、`test_ivf_flat_build_optimization`、`test_opt030_adaptive_ef` 标记为 `#[ignore]`
  3. 最小验证：
     - ✅ `cargo test --test perf_test -q`（5 ignored）
     - ✅ `cargo test --test bench_hnsw_parallel -q`（3 passed, 5 ignored）
     - ✅ `cargo test --lib faiss::ivf_sq_cc::tests::test_ivf_sq_cc_train_add_search`
  4. 结论：默认回归路径进一步剔除性能基准误跑，`BUG-P1-001` 继续收敛；仍待后续轮次确认全量测试全绿。
  状态：BUG-P1-001 In Progress。
- 2026-03-07 14:35: **BUG-P1-001 长跑卡住根因收敛（慢测治理第一轮）**
  1. 同步 baseline：`git fetch origin main` -> `4f60908fc9ad7438b4b8ff64210481ab281009b0`
  2. 根因定位：默认回归路径包含 HNSW 100K 构建性能测试（单测+集成测试各一处），导致 `cargo test` 长尾明显；ScaNN 基础测试数据规模偏大也加剧总时长。
  3. 本轮修复：
     - `src/faiss/hnsw.rs::test_hnsw_build_performance` 标记 `#[ignore]`
     - `tests/opt015_hnsw_build.rs::test_hnsw_build_performance` 标记 `#[ignore]`
     - `src/faiss/scann.rs::test_scann_basic` 数据规模 `n: 1000 -> 256`
  4. 最小验证：
     - ✅ `cargo test --lib test_scann_basic`
     - ✅ `cargo test --tests opt015_hnsw_build`（默认忽略，退出成功）
  5. 结论：已完成“长跑根因定位”子项，BUG-P1-001 剩余“全量回归全绿”验收。
  状态：BUG-P1-001 In Progress。
- 2026-03-07 14:30: **PARITY-P1-009 收敛验收（Index trait + 参数对齐）**
  1. 同步 baseline：`git fetch origin main`（基线仍为 `4f60908fc9ad7438b4b8ff64210481ab281009b0`）
  2. 复核结果：
     - `src/index/minhash_lsh_index_trait.rs`：MinHashLSH 已提供统一 `Index` trait wrapper（含 `create_ann_iterator` / `get_vector_by_ids` / `save` / `load`）
     - `src/api/index.rs`：`mh_*` 参数别名反序列化已对齐（`mh_element_bit_width` / `mh_lsh_band` / `mh_lsh_aligned_block_size` / `mh_lsh_shared_bloom_filter` / `mh_lsh_bloom_false_positive_prob`）
     - `src/ffi.rs`：`MinHashLSH` 类型声明、`create_ann_iterator` 接线、`add_binary` 字节对齐校验均已补齐
  3. 最小验证：
     - ✅ `cargo test --lib minhash_index_trait`
     - ✅ `cargo test --lib test_minhash_cpp_param_aliases_deserialize`
     - ✅ `cargo test --lib ffi::tests::test_index_type_minhash_lsh`
     - ✅ `cargo test --lib ffi::tests::test_minhash_add_binary_rejects_invalid_dim_alignment`
  4. 结论：PARITY-P1-009 验收完成；全量回归失败项归属 BUG-P1-001，不再阻塞本条目关闭。
  状态：PARITY-P1-009 Done。
- 2026-03-07 14:25: **PARITY-P1-009 MinHash FFI 参数语义补齐（维度对齐校验）**
  1. 执行同步与基线更新：`git fetch origin && git rev-parse HEAD && git rev-parse origin/main` -> `4f60908fc9ad7438b4b8ff64210481ab281009b0`
  2. 本轮变更：
     - `src/ffi.rs`：`IndexWrapper::add_binary` 的 MinHash 分支由硬编码参数改为按 `dim(bits)` 推导 `vector_bytes` 与 `mh_vec_length`
     - 增加输入校验：空输入、`vectors.len() % vector_bytes != 0`、`vector_bytes` 非 `u64` 元素对齐、`mh_vec_length == 0` 均返回 `InvalidArg`
     - 新增回归测试 `ffi::tests::test_minhash_add_binary_rejects_invalid_dim_alignment`
  3. 最小验证：
     - ✅ `cargo test --lib test_minhash_add_binary_rejects_invalid_dim_alignment`
  4. 结论：MinHash `mh_*` 参数/维度语义在 FFI add 路径进一步收敛；任务仍受全量回归门槛限制，维持 In Progress。
  状态：PARITY-P1-009 In Progress。
- 2026-03-07 14:25: **PARITY-P1-008 Sparse iterator/filter 行为统一**
  1. 执行同步与基线更新：`git fetch origin && git checkout main && git pull origin main && git rev-parse origin/main` -> `4f60908fc9ad7438b4b8ff64210481ab281009b0`
  2. 本轮变更：
     - `src/faiss/sparse_inverted.rs` 新增统一 helper `ann_results_from_sparse_query`，集中处理 bitset 转换 + iterator 全量候选检索入口
     - `SparseInvertedIndex::create_ann_iterator` 改为通过统一 helper（TAAT 路径）
     - `src/faiss/sparse_wand.rs` 的 `create_ann_iterator` 改为复用统一 helper（WAND 路径）
     - 新增回归测试 `test_sparse_wand_iterator_and_search_with_bitset_consistent`
  3. 最小验证：
     - ✅ `cargo test --lib sparse_inverted::tests::test_sparse_inverted_ann_iterator_respects_bitset`
     - ✅ `cargo test --lib sparse_wand::tests::test_sparse_wand_iterator_and_search_with_bitset_consistent`
  4. 结论：PARITY-P1-008 最后子项“统一 iterator/filter 行为”完成；Sparse Index trait 统一接口任务收敛。
  状态：PARITY-P1-008 Done。
- 2026-03-07 13:15: **BUG-P1-001 全量回归阻塞定位 + 双模块复核（src/index.rs / src/ffi.rs）**
  1. 执行同步与基线更新：`git fetch origin && git checkout main && git pull origin main && git rev-parse origin/main` -> `4f60908fc9ad7438b4b8ff64210481ab281009b0`
  2. C++/Rust 增量扫描：
     - `python3` 关键词统计：`CPP src/index=240`、`CPP include/knowhere/index=53`
     - Rust 关键词统计：`sparse_inverted.rs=3`、`sparse_wand.rs=3`、`index.rs=14`、`ffi.rs=64`
  3. 逐接口复核：
     - `src/index.rs`：`Index` trait 仍提供 `create_ann_iterator/get_vector_by_ids/save/load` 统一入口
     - `src/ffi.rs`：Sparse 仍未新增 `create_ann_iterator`/`save/load` 的特化桥接（依赖索引实现）
  4. 验证：
     - ⏳ `cargo test -q` 在 `501 tests` 场景下长时间无退出（已越过 435/501，且出现 `hnsw_build_performance` / `scann_basic` / `sparse_inverted::test_wand_search_basic` 超 60s 提示）
     - ⚠️ 本轮未拿到完整退出码，无法确认“全量回归恢复”验收
  5. 新发现差距：全量回归除既有失败外，新增“长跑卡住/超时不可判定”阻塞点，需在 BUG-P1-001 下拆分 `hang root-cause` 子任务。
  状态：BUG-P1-001 维持 In Progress（从“失败用例修复”转入“长跑阻塞定位”）。
- 2026-03-07 11:55: **BUG-P1-001 子项修复：IVF-CC 检索候选不足导致 topk 回归失败 + 双模块复核（src/index.rs / src/ffi.rs）**
  1. 执行同步与基线更新：`git fetch origin && git checkout main && git pull origin main && git rev-parse origin/main` -> `4f60908fc9ad7438b4b8ff64210481ab281009b0`
  2. C++/Rust 增量扫描：`find .../knowhere/src/index -maxdepth 2 -type f | wc -l`=49，`find .../include/knowhere/index -maxdepth 2 -type f | wc -l`=8，`find .../knowhere-rs/src/faiss -maxdepth 1 -type f | wc -l`=47
  3. 逐接口复核：
     - `src/index.rs`：`Index` trait 仍提供 `create_ann_iterator/get_vector_by_ids/save/load` 统一入口
     - `src/ffi.rs`：未新增 Sparse 的 `create_ann_iterator` / `save/load` 特化分支（仍依赖索引实现）
  4. 本轮修复：
     - `src/faiss/ivf_flat_cc.rs`：修复 `nprobe` 仅扫描少量倒排桶导致 `top_k` 候选不足的问题，新增按 centroid distance 的 fallback 扫描路径；`num_visited` 改为截断前候选数
     - `src/faiss/ivf_sq_cc.rs`：同样补齐 fallback 扫描逻辑，避免 `test_ivf_sq_cc_train_add_search` 偶发/稳定失败
  5. 验证：
     - ✅ 定向：`cargo test -q faiss::ivf_flat_cc::tests::test_ivf_flat_cc_train_add_search`
     - ✅ 定向：`cargo test -q faiss::ivf_sq_cc::tests::test_ivf_sq_cc_train_add_search`
     - ✅ 定向：`cargo test -q quantization::kmeans::tests::test_kmeans_convergence`
     - ⏳ 全量：`cargo test --tests` 已启动长跑（501 tests，运行中），本轮结束前未获得完整退出状态
  6. 新发现差距：Sparse 模块 `create_ann_iterator` 与 `save/load` 仍未对齐 C++ 能力，维持 P1 缺口。
  状态：BUG-P1-001 从“3 个已知失败点”收敛至“2 个 IVF-CC 子项已修复，待全量回归最终确认”。
- 2026-03-07 11:00: **FFI index_type 返回表补齐（多索引）+ 双模块复核（src/index.rs / src/ffi.rs）**
  1. 执行同步与基线更新：`git fetch origin && git checkout main && git pull origin main && git rev-parse origin/main` -> `7d72f3b9ea4175af10f914fc386528a64a2cff80`
  2. C++/Rust 增量扫描：`find .../knowhere/src/index -maxdepth 1 -type f | wc -l`=7，`find .../include/knowhere/index -maxdepth 1 -type f | wc -l`=8，`find .../knowhere-rs/src/faiss -maxdepth 1 -type f | wc -l`=47
  3. 逐接口复核：
     - `src/index.rs`：`Index` trait 契约入口仍为 `create_ann_iterator/get_vector_by_ids/has_raw_data/save/load`
     - `src/ffi.rs`：`IndexWrapper::index_type` 与 `knowhere_get_index_type` 返回表此前仅覆盖 Flat/HNSW/ScaNN/MinHashLSH
  4. 本轮修复：
     - `src/ffi.rs`：扩展 `IndexWrapper::index_type`，补齐 `HNSW_PRQ/IVF_RABITQ/HNSW_SQ/HNSW_PQ/BinFlat/BinaryHNSW/IVF_SQ8/BinIVFFlat/SparseWand/SparseWandCC`
     - `src/ffi.rs`：同步扩展 `knowhere_get_index_type` 静态 C 字符串返回表，消除 Unknown 回退误判
     - `src/ffi.rs`：新增回归测试 `test_index_type_hnsw_pq`、`test_index_type_sparse_wand`，并抽出 `assert_index_type`
  5. 验证：
     - ✅ 定向：`cargo test -q test_index_type_`（5 passed）
     - ❌ 全量：`cargo test -q` / `cargo test --lib -q` / `cargo test --tests -q` 仍失败（观测到 `ivf_sq_cc`、`ivf_flat_cc`、`quantization::kmeans` 等既有失败）
  6. 新发现差距：全量回归失败点从“泛化失败”可定位到 `ivf_sq_cc` / `ivf_flat_cc` / `quantization::kmeans`，需独立拆分稳定性修复任务。
  状态：FFI 类型声明一致性进一步提升，整体模块状态维持 Partial（受全量回归阻断）。
- 2026-03-07 08:55: **MinHashLSH FFI 声明一致性修复 + 双模块复核（src/index.rs / src/ffi.rs）**
  1. 执行同步与基线更新：`git fetch origin && git checkout main && git pull origin main && git rev-parse origin/main` -> `7d72f3b9ea4175af10f914fc386528a64a2cff80`
  2. C++/Rust 增量扫描：`find .../knowhere/src/index -maxdepth 1 -type f | wc -l`=7，`find .../include/knowhere/index -maxdepth 1 -type f | wc -l`=8，`find .../knowhere-rs/src/faiss -maxdepth 1 -type f | wc -l`=47
  3. 逐接口复核：
     - `src/index.rs`：`trait Index` 仍包含 `create_ann_iterator`/`get_vector_by_ids`/`has_raw_data` 统一入口
     - `src/ffi.rs`：`IndexWrapper::index_type` 已含 `MinHashLSH`；`knowhere_get_index_type` 返回表此前缺该分支
  4. 本轮修复：
     - `src/ffi.rs`：为 `knowhere_get_index_type` 新增 `"MinHashLSH"` 分支，消除类型字符串不一致
     - `src/ffi.rs`：为 `IndexWrapper::create_ann_iterator` 新增 `minhash_lsh` 分支，统一 ANN iterator 入口
     - `src/ffi.rs`：新增回归测试 `ffi::tests::test_index_type_minhash_lsh`
  5. 验证：
     - ✅ 定向：`cargo test -q ffi::tests::test_index_type_minhash_lsh`
     - ❌ 全量：`cargo test -q` / `cargo test --lib -q` / `cargo test --tests -q` 仍失败（AISAQ 与 ScaNN/HNSW 既有失败）
  6. 新发现差距：MinHash FFI 已接线，但全量回归仍被非 MinHash 模块阻断，P1 回归债未清。
  状态：MinHash 模块维持 Partial（FFI 接线已补齐，待全量回归恢复）。
- 2026-03-06 23:58: **BUG-P0-004 编译回归收敛 + 双模块复核（src/index.rs / src/ffi.rs）**
  1. 执行同步与基线更新：`git fetch origin && git checkout main && git pull origin main && git rev-parse origin/main` -> `388aea90260084965f965b29e0a8b87b7a808d51`
  2. C++/Rust 增量扫描：`ls .../knowhere/src/index | wc -l`=17，`ls .../include/knowhere/index | wc -l`=8，`ls .../knowhere-rs/src/faiss | wc -l`=47
  3. 深比对（本轮按要求至少 2 模块逐接口复核）：
     - `src/index.rs`：`trait Index` 统一契约仍在 (`src/index.rs:127`)
     - `src/ffi.rs`：`CIndexConfig` / `knowhere_get_index_type` / `knowhere_create_ann_iterator` 接口声明完整
  4. 回归修复：批量修复 tests/examples 的 `IndexConfig::data_type` 缺失；统一 `crate::api::DataType` -> `knowhere_rs::api::DataType`
  5. 验证结果：
     - `cargo test --tests --no-run -q` ✅（data_type 迁移导致的编译错误已清零）
     - `cargo test -q` / `cargo test --lib -q` ❌（运行期失败：AISAQ trait tests、ScaNN FFI tests、kmeans convergence 等）
  6. 新发现差距：从“编译回归”转为“运行期功能回归”聚焦，下一步应拆分 AISAQ/ScaNN/KMeans 的稳定性缺口。
  状态：BUG-P0-004 关闭（编译回归已修复）；全量功能回归仍为 P1。
- 2026-03-06 18:02: **MinHash Index trait wrapper 接入 + 全量测试回归诊断**
  1. 执行同步与基线更新：`git fetch origin && git checkout main && git pull origin main && git rev-parse origin/main` -> `388aea90260084965f965b29e0a8b87b7a808d51`
  2. 对齐扫描命令：`ls /Users/ryan/Code/vectorDB/knowhere/src/index && ls /Users/ryan/Code/vectorDB/knowhere/include/knowhere/index && ls /Users/ryan/.openclaw/workspace-builder/knowhere-rs/src/faiss`
  3. 逐文件复核：`src/index.rs`、`src/ffi.rs`、`src/index/minhash_lsh.rs`、`src/index/minhash_lsh_index_trait.rs`、`src/index/minhash/minhash_index_node.cc`
  4. 变更：新增 `src/index/minhash_lsh_index_trait.rs`，为 MinHashLSHIndex 实现统一 `Index` trait（train/add/search/range/get_vector/serialize/iterator 元数据）并补齐 5 个单测；`src/index.rs` 导出模块。
  5. 验证：`cargo test --lib minhash_lsh_index_trait` 通过；`cargo test` / `cargo test --tests` 失败，失败原因为 tests 侧 `IndexConfig.data_type` 迁移未完成（非本轮 MinHash 模块逻辑错误）。
  6. 新发现差距：全量测试编译回归，需新增 P0 修复任务 `BUG-P0-004`。
  状态：MinHash 模块从 Partial（无 trait）提升到 Partial（trait 已接入，仍待 FFI 统一接线与全量回归恢复）。
- 2026-03-06 14:38: **MinHash FFI 查询长度对齐修复 + 抽样复核**
  1. 执行同步与基线更新：`git fetch origin && git checkout main && git pull origin main && git rev-parse origin/main` -> `388aea90260084965f965b29e0a8b87b7a808d51`
  2. 抽样深比对命令：`ls /Users/ryan/Code/vectorDB/knowhere/src/index && ls /Users/ryan/Code/vectorDB/knowhere/include/knowhere/index && ls /Users/ryan/.openclaw/workspace-builder/knowhere-rs/src/faiss`
  3. 复核接口：`src/index.rs`、`src/ffi.rs`、`src/index/minhash_lsh.rs`、`src/ffi/minhash_lsh_ffi.rs`
  4. 修复项：`src/index/minhash_lsh.rs` 新增 `vector_byte_size()`；`src/ffi/minhash_lsh_ffi.rs` 将 query/queries 长度计算从占位逻辑改为 `mh_vec_length * mh_vec_element_size`
  5. 新增回归测试：`test_search_uses_vector_byte_size`
  状态：MinHash 模块维持 Partial（Index trait wrapper 仍缺失），但 FFI query 长度缺口已关闭，风险从 P1-high 降为 P1-medium。
- 2026-03-06 13:32: **MinHash 参数别名对齐 + 目录级复核**
  1. 执行同步与基线更新：`git fetch origin && git checkout main && git pull origin main && git rev-parse origin/main` -> `388aea90260084965f965b29e0a8b87b7a808d51`
  2. 扫描 C++ 目录：`src/index/`、`include/knowhere/index/`；并对照 Rust `src/index.rs`、`src/ffi.rs`、`src/index/minhash_lsh.rs`
  3. MinHash 参数命名对齐（部分完成）：`src/api/index.rs` 新增 `mh_*` 到 Rust 参数字段的 serde alias + 单测
  4. 新发现差距：MinHash 仍未接入统一 `Index trait`；`src/ffi/minhash_lsh_ffi.rs` 的 query 大小计算仍为占位逻辑（`count()*count()`）
  状态：MinHash 模块保持 Partial，风险维持 P1。
- 2026-03-06 12:32: **AISAQ Index trait 实现** - 为 AisaqIndex 实现完整的 Index trait，包括：
  1. 基础生命周期方法：train/add/search/search_with_bitset/save/load/serialize_to_memory/deserialize_from_memory
  2. 高级接口：AnnIterator (AisaqAnnIterator) / get_vector_by_ids / has_raw_data
  3. 元数据方法：index_type/dim/count/is_trained/has_raw_data
  4. 添加 Serialize/Deserialize 到 AisaqConfig
  5. 创建测试套件验证实现（5 个测试）
  状态：AISAQ 模块从 Partial 升级为 Done（Index trait 实现完成）。
- 2026-03-06 10:32: **ScaNN Index trait 验证** - 确认 ScaNNIndex 已实现完整 Index trait，包括：
  1. 基础生命周期方法：train/add/search/search_with_bitset/save/load
  2. 高级接口：get_vector_by_ids（支持但需检查 has_raw_data）/has_raw_data（取决于 reorder_k）/create_ann_iterator
  3. 元数据方法：index_type/dim/count/is_trained/has_raw_data
  4. 创建测试套件验证实现（6 个测试全部通过）
  5. 修复编译错误：binary_hnsw.rs 和 diskann.rs 的 data_type 字段问题
  状态：ScaNN 模块从 Partial 升级为 Done（Index trait 实现完成并测试验证）。
- 2026-03-06 07:35: **DiskANN Index trait 实现** - 为 DiskAnnIndex 实现完整的 Index trait，包括：
  1. 基础生命周期方法：train/add/search/range_search/save/load
  2. 高级接口：AnnIterator (DiskAnnIteratorWrapper) / get_vector_by_ids
  3. 元数据方法：index_type/dim/count/is_trained/has_raw_data
  4. 创建测试套件验证实现（test_diskann_index_trait）
  状态：DiskANN 模块从 Partial 升级为 Done（Index trait 实现完成）。
- 2026-03-06 06:35: **IVF 系列架构缺口修复** - 为 IvfSq8Index 和 IvfRaBitqIndex 实现完整的 Index trait，包括：
  1. 基础生命周期方法：train/add/search/search_with_bitset/save/load
  2. 高级接口：AnnIterator（两个索引）/ get_vector_by_ids（仅 IVF-SQ8，IVF-RaBitQ 因有损压缩返回 Unsupported）
  3. 元数据方法：index_type/dim/count/is_trained/has_raw_data
  4. 创建测试套件验证实现（7 个测试全部通过）
  状态：IVF core 模块从 Partial 升级为 Done（Index trait 实现完成），剩余参数校验统一化任务。
- 2026-03-06 05:35: **IVF 系列架构缺口诊断** - 发现 IVF-SQ8/IVF-RaBitQ 未实现 Index trait，仅通过 FFI IndexWrapper enum dispatch 访问。这意味着：
  1. IVF 系列无法通过统一 Index trait 调用高级接口（AnnIterator/get_vector_by_ids）
  2. FFI 层需要为每种 IVF 类型重复实现调用逻辑
  3. 参数校验和错误处理可能不一致
  行动：将 PARITY-P1-002 升级为关键架构任务，需要为 IVF 系列实现 Index trait wrapper。
- 2026-03-06 04:35: **HNSW 高级路径测试** - 创建 `tests/test_hnsw_advanced_paths.rs`，覆盖 get_vector_by_ids、AnnIterator、serialize/deserialize、range_search（Unsupported）。5 个测试全部通过。PARITY-P1-001 完成，HNSW 模块状态升级为 Partial → Done（高级路径）。
- 2026-03-06 03:35: **核心契约一致性验证** - 验证所有索引对未实现方法的错误处理一致：Index trait 提供默认 Unsupported 实现；FFI 层 19 处 NotImplemented 返回；所有非 GPU 索引行为一致。核心契约状态从 Partial 升级为 Done（P0 降级）。
- 2026-03-06 01:35: **实现 FFI AnnIterator 接口** - 添加 `knowhere_create_ann_iterator`/`knowhere_ann_iterator_next`/`knowhere_free_ann_iterator` 三个 FFI 函数，支持 HNSW/ScaNN/HNSW-PQ 索引
- 2026-03-06 00:35: **更新 FFI 能力矩阵** - 标记 HNSW/ScaNN/HNSW-PQ/DiskANN 的 AnnIterator 为 ✅；HNSW GetByID ✅；ScaNN GetByID ⚠️；DiskANN GetByID ⚠️
- 2026-03-05 23:35: **实现 AnnIterator 接口** - HNSW, ScaNN, HNSW-PQ, DiskANN 四个索引实现 create_ann_iterator，验收标准达成（>=3个索引）
- 2026-03-05 22:32: 详细对比 C++/Rust 核心接口，发现 IsAdditionalScalarSupported/GetIndexMeta 缺失，AnnIterator 未实现
- 2026-03-05 21:22: 扫描 C++/Rust 接口对齐状态，确认 AnnIterator 未实现但已定义，HNSW 实现核心接口
- 2026-03-05 20:40: 添加 AnnIterator 接口定义，创建 FFI 能力矩阵文档
- 2026-03-05 20:35: 确认 ivf_sq_cc 所有测试通过 (6/6)，BUG-P0-003 完成
- 2026-03-05 19:35: 修复 3 个 P0 BUG (mini_batch_kmeans/diskann_complete/ivf_sq_cc SIMD 切片长度问题)

## 1. Scope

- In scope: non-GPU parity against C++ knowhere
- Out of scope: GPU/cuVS implementation parity

## 2. Status Legend

- `Done`: implementation and behavior aligned
- `Partial`: implemented but behavioral/edge mismatch remains
- `Blocked`: intentionally deferred or requires prerequisite
- `Missing`: not implemented

Risk levels:

- `P0`: blocks production parity
- `P1`: important functional/behavioral gap
- `P2`: optimization/documentation/coverage gap

## 3. Module-Level File Mapping and Gap Items

| Module | Native file(s) | Rust file(s) | Status | Risk | Pending interface items |
|---|---|---|---|---|---|
| Core contract | `include/knowhere/index/index.h`, `include/knowhere/index/index_node.h` | `src/index.rs`, `src/api/search.rs` | Done | P1 | ✅ lifecycle contract unified (2026-03-06); ✅ AnnIterator trait implemented (2026-03-05); all indexes return consistent Unsupported for unimplemented methods |
| Index factory/legality | `include/knowhere/index/index_factory.h`, `include/knowhere/index/index_table.h`, `include/knowhere/comp/knowhere_check.h` | `src/api/index.rs`, `src/api/data_type.rs`, `src/api/legal_matrix.rs`, `src/ffi.rs` | Done | P2 | ✅ centralized legal matrix 已实现；✅ `IndexConfig::validate()` 与 FFI `IndexWrapper::new()` 均走统一校验入口；✅ 当前已无运行时代码缺口，旧 `Partial` 仅为历史残留文案 |
| HNSW | `src/index/hnsw/faiss_hnsw.cc` | `src/faiss/hnsw.rs` | Done | P1 | ✅ AnnIterator (2026-03-05), ✅ get_vector_by_ids (2026-03-05), ✅ serialize/deserialize, ✅ range_search (Unsupported, tested 2026-03-06); all advanced paths tested and aligned |
| IVF core | `src/index/ivf/ivf.cc`, `src/index/ivf/ivf_config.h` | `src/faiss/ivf.rs`, `src/faiss/ivf_flat.rs`, `src/faiss/ivfpq.rs`, `src/api/index.rs` | Done | P1 | ✅ Index trait implemented for IvfSq8Index and IvfRaBitqIndex (2026-03-06); ✅ AnnIterator; ✅ get_vector_by_ids (IVF-SQ8 only); parameter coverage and edge behavior alignment remaining; SIMD slice fix in ivf_sq_cc (2026-03-05) |
| RaBitQ | `src/index/ivf/ivfrbq_wrapper.*` | `src/faiss/ivf_rabitq.rs`, `src/faiss/rabitq_ffi.rs` | Done | P1 | ✅ Index trait implemented (2026-03-06); ✅ AnnIterator; ⚠️ get_vector_by_ids (Unsupported for lossy compression); query-bits and config boundary consistency |
| DiskANN | `src/index/diskann/diskann.cc`, `src/index/diskann/diskann_config.h` | `src/faiss/diskann.rs`, `src/faiss/diskann_complete.rs` | Done | P1 | ✅ Index trait implemented (2026-03-06); ✅ AnnIterator (DiskAnnIteratorWrapper); ✅ get_vector_by_ids; lifecycle parity and config semantics; add_batch SIMD slice fix (2026-03-05) |
| AISAQ | `src/index/diskann/diskann_aisaq.cc`, `src/index/diskann/aisaq_config.h` | `src/faiss/diskann_aisaq.rs`, `src/faiss/aisaq.rs` | Done | P1 | ✅ Index trait implemented (2026-03-06); ✅ AnnIterator; ✅ get_vector_by_ids; parameter and file-layout behavior alignment |
| ScaNN | - | `src/faiss/scann.rs` | Done | P1 | ✅ AnnIterator (2026-03-05), ✅ get_vector_by_ids (2026-03-06), ✅ has_raw_data (depends on reorder_k), ✅ Index trait (2026-03-06); tested |
| HNSW-PQ | - | `src/faiss/hnsw_pq.rs` | Done | P2 | ✅ AnnIterator (2026-03-05); ✅ `has_raw_data=false`（lossy PQ）; ✅ `get_vector_by_ids` 显式稳定返回 Unsupported；✅ `save/load` 当前在 persistence scope 内显式稳定返回 Unsupported |
| Sparse | `src/index/sparse/sparse_index_node.cc`, `src/index/sparse/sparse_inverted_index.h` | `src/faiss/sparse_inverted.rs`, `src/faiss/sparse_wand.rs`, `src/faiss/sparse_wand_cc.rs` | Done | P1 | ✅ `SparseInverted`/`SparseWand` 已接入 `create_ann_iterator` + `save/load`；✅ bitset + iterator + persistence 回归已覆盖；ℹ️ `SparseWandCC` 仍为并发包装层，当前未纳入统一 `Index` trait / 持久化承诺（不影响本模块 parity 收口） |
| MinHash | `src/index/minhash/minhash_index_node.cc`, `src/index/minhash/minhash_lsh_config.h` | `src/index/minhash_lsh.rs`, `src/index/minhash_lsh_index_trait.rs`, `src/ffi/minhash_lsh_ffi.rs`, `src/api/index.rs` | Done | P1 | ✅ 参数别名映射已补齐；✅ FFI query 长度已对齐 `mh_vec_length * mh_vec_element_size`；✅ `MinHashLSHIndex` 已接入 `Index trait`；✅ `knowhere_get_index_type`/`create_ann_iterator` 已补齐 MinHashLSH 分支；✅ `add_binary` 按 `dim(bits)` 做字节/对齐校验；全量回归问题已归属 BUG-P1-001 |
| FFI ABI | C++ factory + index runtime behavior | `src/ffi.rs`, `docs/FFI_CAPABILITY_MATRIX.md` | Done | P1 | ✅ capability matrix documented; ✅ consistent error handling (19 NotImplemented returns); ✅ `IsAdditionalScalarSupported` / `GetIndexMeta` 统一入口已补齐并具备回归覆盖 |

## 4. Validation Policy

- Every benchmark must report:
  - ground truth source
  - R@10
  - QPS
  - credibility tag (`trusted` / `unreliable` / `recheck required`)
- Credibility rules:
  - R@10 >= 80% => trusted (if setup valid)
  - 50% <= R@10 < 80% => unreliable
  - R@10 < 50% => recheck required

### Cross-dataset artifact 记录模板（BENCH-P2-003）

后续轮次引用 `benchmark_results/cross_dataset_sampling.json` 时，最少记录以下字段：
- dataset / index / base_size / query_size / dim
- params / ground_truth_source
- recall_at_10 / qps / confidence / runtime_seconds

## 5. Audit Changelog

- 2026-03-08 20:28: **OPT-P2-004 计划前复核：Index factory/legality 已满足 Done，只剩治理状态漂移待收口**
  - 复核文件：`src/api/index.rs`、`src/api/legal_matrix.rs`、`src/ffi.rs`、`TASK_QUEUE.md`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md`。
  - 结论：`IndexConfig::validate()` 已统一调用 legality matrix；FFI `IndexWrapper::new()` 也会在构造期执行 `validate_index_config`，非法 `(index_type, data_type, metric)` 组合会在入口被拒绝，而不是流入运行时。
  - 治理判断：审计表中 `Index factory/legality=Partial` 已不再对应真实代码缺口，属于 `PARITY-P1-004` 落地后的历史残留；本轮 plan 应将其提升为当前治理任务而不是继续制造新的功能开发任务。
  - 下一步：提升 `OPT-P2-004` 为当前 TODO，收敛 queue/roadmap/gap/audit 的状态表述。

- 2026-03-08 19:35: **PARITY-P1-011 收口复核：FFI ABI 元数据契约已具备统一入口**
  - 复核文件：`src/index.rs`、`src/ffi.rs`、`include/knowhere/index/index.h`、`include/knowhere/index/index_node.h`。
  - 结论：Rust 侧已补齐 `Index::is_additional_scalar_supported` / `Index::get_index_meta` 最小抽象；FFI 暴露 `knowhere_is_additional_scalar_supported`、`knowhere_get_index_meta`、`knowhere_free_cstring`。
  - 运行时语义：当前统一采用保守 contract——附加标量能力默认 `false`，`GetIndexMeta` 返回稳定 JSON summary（`index_type/dim/count/is_trained/has_raw_data/additional_scalar_supported`），避免 capability 声明与实际调用错位。
  - 回归证据：`ffi::tests::test_ffi_abi_metadata_contract`、`ffi::tests::test_index_type_*`、`ffi::tests::test_minhash_add_binary_rejects_invalid_dim_alignment`。
- 2026-03-08 19:10: **PARITY-P1-010 收口复核：Sparse 高级接口已具备可验证支持矩阵**
  - 复核文件：`src/faiss/sparse_inverted.rs`、`src/faiss/sparse_wand.rs`、`src/index.rs`。
  - 结论：`SparseInverted`/`SparseWand` 均已通过统一 `Index` trait 暴露 `create_ann_iterator`、`save`、`load`；底层持久化使用 `SparseInvertedSnapshot`（`bincode` + version gate），不再是 Unsupported。
  - 回归证据：`test_sparse_inverted_ann_iterator_respects_bitset`、`test_sparse_wand_iterator_and_search_with_bitset_consistent`、`test_sparse_inverted_save_load_roundtrip_preserves_iterator_and_vectors`、`test_sparse_wand_save_load_roundtrip_preserves_wand_behavior`。
  - 受限边界：`SparseWandCC` 当前仅为并发包装层，未承诺统一 `Index` trait / persistence parity；本轮 P1 收口范围限定为 `SparseInverted`/`SparseWand`。
- 2026-03-07 00:38: 增量审计 Sparse 模块（`src/index/sparse/*` vs `src/faiss/sparse_inverted.rs` / `src/faiss/sparse_wand.rs` / `src/index.rs` / `src/ffi.rs`）。
  - ✅ SparseInverted/SparseWand 已通过 Index trait 暴露 train/add/search/search_with_bitset/get_vector_by_ids。
  - （历史记录）当时观察到 `save/load` / `create_ann_iterator` 尚未形成闭环；该结论已由 2026-03-08 19:10 复核覆盖。 
- 2026-03-05 22:32: Detailed interface comparison between C++ and Rust.
  - **C++ Index class methods (17 total):**
    - Build/Train/Add/Search (core lifecycle) ✅
    - AnnIterator (streaming results) ⚠️ trait defined but not implemented in any index
    - RangeSearch (radius-based search) ⚠️ some indexes return Unsupported
    - GetVectorByIds (vector retrieval) ⚠️ some indexes return Unsupported
    - HasRawData (raw data check) ✅
    - IsAdditionalScalarSupported ❌ **MISSING** in Rust
    - GetIndexMeta ❌ **MISSING** in Rust
    - Serialize/Deserialize (BinarySet) ✅ (serialize_to_memory/deserialize_from_memory)
    - DeserializeFromFile ⚠️ Rust has save/load but not exact equivalent
    - Dim/Size/Count/Type ✅ (dim/count/index_type; missing Size)
  - **Priority actions:**
    - P0: Implement AnnIterator for core indexes (HNSW/IVF/Flat)
    - P1: Add IsAdditionalScalarSupported and GetIndexMeta methods
    - P1: Ensure all indexes properly implement or reject unsupported methods
  - **Files checked:**
    - C++: `/Users/ryan/Code/vectorDB/knowhere/include/knowhere/index/index.h:152-236`
    - Rust: `/Users/ryan/.openclaw/workspace-builder/knowhere-rs/src/index.rs:102-267`

- 2026-03-05 21:22: Scanned interface alignment between C++ and Rust.
  - C++ Index class methods: Build/Train/Add/Search/AnnIterator/RangeSearch/GetVectorByIds/HasRawData/Serialize/Deserialize/DeserializeFromFile
  - Rust Index trait methods: train/add/search/range_search/create_ann_iterator/get_vector_by_ids/has_raw_data/serialize_to_memory/deserialize_from_memory/save/load
  - Gap: AnnIterator defined but not implemented in any index; DeserializeFromFile missing in Rust
  - HNSW implements core methods (train/add/search/range_search/get_vector_by_ids/save/load)
  - Next: Verify all core indexes implement or reject unsupported methods consistently

- 2026-03-05: Initialized parity audit baseline with module/file mapping and risk triage.

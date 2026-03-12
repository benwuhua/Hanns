# Knowhere-RS Gap Analysis (Non-GPU)

Last updated: 2026-03-12 09:29 UTC
Scope: Non-GPU production parity against C++ knowhere

## 1. Baseline and Method

- Rust repo: `/Users/ryan/.openclaw/workspace-builder/knowhere-rs`
- C++ repo: `/Users/ryan/Code/vectorDB/knowhere`
- Source of truth for parity status: `docs/PARITY_AUDIT.md`
- Priority order: BUG > CORE(IMPL/PERF) > SEMANTIC/PROD > BENCH

Evaluation dimensions:

1. Functional parity (core index lifecycle and query behavior)
2. Interface parity (Build/Train/Add/Search/RangeSearch/AnnIterator/GetVectorByIds/Serialize/Deserialize)
3. Validation quality (ground truth, recall constraints, regression tests)
4. Production readiness (stability, configurability, observability)

## 2. Current High-Level Gaps

## P0 (Critical)

- ✅ No active P0 blocker in current queue snapshot.

## P1 (Important)

- ✅ 当前无活跃 P1 parity blocker；入口级 parity 已基本收口。
- 但“入口存在”不等于“原生语义完全对齐”，下一阶段主缺口已从 P1 parity closure 转移到 semantic fidelity。

## P2 (Optimization / Hardening)

- ✅ 当前已无活跃 P2 治理缺口；`HNSW-PQ` 与 `Index factory/legality` 两个尾部历史项均已收口为 Done。
- ✅ 分层门禁执行入口已收口：`scripts/gate_profile_runner.sh` 已固化 default/full/long-tests，并与 `memory/*_RESULT.json.gate_profile` 建立一一映射。
- ✅ recall-gated baseline 覆盖面缺口已收口：已覆盖 ScaNN/RaBitQ/Sparse，并统一输出 `confidence_explanation`。
- ✅ cross-dataset（SIFT/GIST/随机）抽样已形成稳定 artifact 流水线（`benchmark_results/cross_dataset_sampling.json`）。

## P3 (Core Implementation / Semantic Fidelity / Production Readiness / Performance Advantage)

- 🔄 `HNSW-REOPEN-001`: round 3 is now active.
  - `benchmark_results/hnsw_p3_002_final_verdict.json` remains the current historical truth: HNSW is still `functional-but-not-leading`.
  - `benchmark_results/hnsw_reopen_round2_authority_summary.json` still records the round-2 hard stop: the authority lane moved from `710.962` down to `521.031` qps while native BF16 stayed near `10519.683`.
  - `benchmark_results/hnsw_reopen_round3_baseline.json` now freezes that hard-stop outcome as the starting point for a narrower round-3 hypothesis: `distance_compute_inner_loop`.
  - `benchmark_results/hnsw_reopen_distance_compute_profile_round3.json` now includes the first round-3 rework refresh: aggregate `distance_compute` moved from `40.165ms` down to `38.528ms`, the dominant `layer0_query_distance` bucket moved from `32.500ms` down to `31.244ms`, `upper_layer_query_distance` also dropped to `7.284ms`, and sample-search qps rose from `2023.694` to `2069.930`.
  - The current active question is therefore no longer “can the layer-0 L2 `query -> node` path be specialized?” but “does this synthetic/profile gain survive the real same-schema authority lane?”
  - The next tracked feature is `hnsw-round3-authority-same-schema-rerun`.
- ✅ `CORE-P0-001`: 远端 x86 SIMD 验证链已恢复可执行并取得新鲜证据。最新复核中，本地 `cargo test --lib -q`、远端 `cargo test --features simd simd::tests -- --nocapture`、远端 `cargo test --lib --features simd test_x86_simd_l2_reduction_matches_scalar_on_irregular_input -- --nocapture` 均已通过；`default+simd` 不再因 toolchain/脚本漂移阻断后续核心路径工作。
- ✅ `HNSW-P1-001`: HNSW 已完成首轮远端 before/after artifact 落地；当前证据显示 recall 基本持平（`0.217 -> 0.215`）但 qps 大幅提升（`~1621 -> ~19235`），由于 recall 仍低于可信阈值，这条结果已被诚实归档为 `recheck required / no-go`，不再作为当前活动 blocker。
- ✅ `IVFPQ-P1-002`: IVF/PQ 已完成 focused reality audit，并留下 `benchmark_results/ivfpq_p1_002_focused.json` 作为可审计 no-go / recheck-required artifact。结论已明确：`src/faiss/ivf.rs` 是 placeholder coarse-assignment scaffold，真实热点路径在 `src/faiss/ivfpq.rs`。
- ✅ `IVFPQ-P3-001`: `ivfpq-hot-path-audit` 已用本地回归和 authority 日志把上述边界锁定为可执行事实：`src/faiss/ivfpq.rs` 负责 coarse centroid + residual PQ training + encoded inverted-list storage；`src/faiss/ivf.rs` 仍只适合作为简化 scaffold，不再用于 IVF-PQ parity / performance 叙述。
- ✅ `DISKANN-P1-003`: Rust DiskANN 的边界已被诚实收口。`l2_sqr` 的 sqrt round-trip 反模式已修复，且 `PQCode` 已被单测锁定为“按子段均值量化的 placeholder”；本轮 `diskann-scope-boundary-audit` 又把 `src/faiss/diskann.rs` 与 `src/faiss/diskann_aisaq.rs` 的角色分离成可执行事实：前者是简化 Vamana + placeholder PQCode，后者提供真实 flash-layout / page-cache / beam-search skeleton，但两者都不是 native-comparable SSD DiskANN 管线。

- ✅ `SEM-P3-001`: `DiskANN` / `AISAQ` / `HNSW` / `IVF` / `Sparse` / `ScaNN` 的 `GetVectorByIds` / `HasRawData` Phase-5 语义尾项已完成 focused 收敛；当前不再缺“入口存在但边界语义不可解释”的 semantic-fidelity blocker。
- ✅ `ABI-P3-002`: FFI metadata / additional-scalar 已从“最小稳定摘要”升级为逐索引可解释 contract；HNSW / IVF / ScaNN / Sparse 的 capability/semantics/unsupported_reason 已具备 focused FFI 回归覆盖，当前不再是活跃 P3 blocker。
- ✅ `PERSIST-P3-003`: persistence / `DeserializeFromFile` 语义矩阵已系统化，`file_save_load` / `memory_serialize` / `deserialize_from_file` 的 supported / constrained / unsupported 边界现已可审计且具备 focused regressions。
- ✅ `OBS-P3-005`: 最小 runtime governance contract 已收口到 `knowhere_get_index_meta`，统一暴露 `observability` / `trace_propagation` / `resource_contract` 三组字段；当前不再缺“缺少稳定 schema/透传入口/资源口径”的 P3 blocker。
- ✅ `PERF-P3-004`: native benchmark harness 缺口已关闭；远端 x86 现已可构建 `benchmark_float_qps`、执行 `--gtest_list_tests`，且输出字段与 Rust parser 保持兼容。
- ✅ `BASELINE-P3-001`: 可信 native-vs-rs 基线 **已收口**
  - 修复了方法论 bug（recall@100 误标为 recall@10）
  - 验证证明：Rust HNSW 需要 ef=2000 才能达到当前 trusted native ef=139 同等 recall@10≈0.9505
  - 量化差距：QPS 726 vs 15144.811（**20.9x**）at recall@10≥0.95
  - 根因：建图质量差距（邻居选择算法），非搜索路径 bug
  - 证据：`benchmark_results/baseline_p3_001_stop_go_verdict.json`
- ✅ `IVFPQ-P3-003`: **no-go**
  - family-level final verdict 已归档到 `benchmark_results/ivfpq_p3_003_final_verdict.json`
  - `ivfpq_p1_002_focused.json`、`recall_gated_baseline.json`、`cross_dataset_sampling.json` 一致表明 IVF-PQ 仍未越过 `0.8` recall gate，且 confidence 仍非 trusted
  - 默认 benchmark lane 现已改为真实 regressions：`tests/bench_ivf_pq_perf.rs`、`tests/bench_recall_gated_baseline.rs`、`tests/bench_cross_dataset_sampling.rs` 不再是 `0 tests` 壳
- ✅ `DISKANN-P3-004`: **constrained**
  - `benchmark_results/diskann_p3_004_final_verdict.json` 已把 DiskANN family-level final classification 归档为 `constrained`
  - `benchmark_results/diskann_p3_004_benchmark_gate.json` 继续把 benchmark lane 固化为 `no_go_for_native_comparable_benchmark`
  - authority-refreshed `benchmark_results/cross_dataset_sampling.json` 现已包含三个 DiskANN sampled rows，且 recall 仍全部低于 `0.8` gate、confidence 仍为非 trusted
  - 默认 library / benchmark / compare lanes (`cargo test --lib diskann -- --nocapture`, `tests/bench_diskann_1m.rs`, `tests/bench_compare.rs`) 现已共同阻止把这条功能可用但受限的 Vamana/AISAQ 实现误读为 native-comparable DiskANN
- ✅ `FINAL-CORE-CLASSIFICATION`: core CPU paths 的最终分类已形成统一 rollup
  - `benchmark_results/final_core_path_classification.json` 现已统一归档 HNSW=`functional-but-not-leading`、IVF-PQ=`no-go`、DiskANN=`constrained`
  - `tests/bench_recall_gated_baseline.rs` 与 `tests/bench_cross_dataset_sampling.rs` 现已将这一 rollup 锁到现有 authority-backed baseline/cross-dataset artifacts，而不再只靠分散 family verdict docs
- ✅ `FINAL-PERFORMANCE-LEADERSHIP-PROOF`: 最终 performance-leadership criterion 已被明确归档为未满足
  - `benchmark_results/final_performance_leadership_proof.json` 现已把项目级 completion criterion 写成 `criterion_met=false`
  - `tests/bench_hnsw_cpp_compare.rs` 现已将这个结论锁到 HNSW 的 authority-backed same-schema blocker 和 `benchmark_results/final_core_path_classification.json`
  - 当前结论是“项目尚未证明任何一条 core CPU lane 具备可信 leadership”，而不是“证据不足待定”
- ✅ `HNSW-P3-002`: **functional-but-not-leading**
  - layer-0 语义差异、same-schema HDF5 refresh、以及 HNSW FFI / persistence contract 均已 authority 收口
  - 当前 family 级最终结论已归档到 `benchmark_results/hnsw_p3_002_final_verdict.json`：Rust HNSW 具备可信 recall 与生产契约，但在当前 trusted same-schema lane 上 native 仍约快 `14.8x`，因此继续阻止 leadership claim
  - 该 verdict 现已降级为 HNSW reopen line 的历史基线；是否继续保持这一结论，将由新的 authority reopen artifacts 决定
- ✅ `FINAL-PRODUCTION-ACCEPTANCE`: 项目级最终 verdict 已归档为 `not accepted`
  - `benchmark_results/final_production_acceptance.json` 现已明确记录：production engineering gates 全部关闭，但 project-level acceptance 仍为 `false`
  - authority `fmt/clippy/full_regression` gate 已在最终 verdict feature 下复跑通过，且 `tests/test_final_production_acceptance.rs` 已把该 verdict 锁进默认 `cargo test --tests -q` surface
  - 当前 blocker 不是“还没验完”，而是已经归档的事实：`benchmark_results/final_performance_leadership_proof.json` 仍为 `criterion_met=false`，`benchmark_results/final_core_path_classification.json` 仍是 HNSW=`functional-but-not-leading`、IVF-PQ=`no-go`、DiskANN=`constrained`

## 3. Validation Gaps

Required validation policy:

- All performance claims must include ground truth origin.
- If R@10 < 80%, QPS must be marked as "unreliable".
- If R@10 < 50%, benchmark result must be marked as "recheck required".
- Regression gate must be profile-driven and reproducible (`default/full/long-tests`).

## 4. Module Focus (Non-GPU)

Primary modules for closure verification:

- Legality / validation governance: `src/api/index.rs`, `src/api/legal_matrix.rs`, `src/ffi.rs`
- Audit + capability docs: `docs/PARITY_AUDIT.md`, `docs/FFI_CAPABILITY_MATRIX.md`
- Governance sync: `TASK_QUEUE.md`, `DEV_ROADMAP.md`, `GAP_ANALYSIS.md`

## 5. Completion Definition

Historical final artifacts remain explicit, but the repo is no longer in a pure terminal state:

1. `benchmark_results/final_production_acceptance.json` 仍然是当前项目级历史结论：`not accepted on current remote x86 evidence`。
2. 当前唯一被重新打开的工作线仍然是 HNSW；future work must improve HNSW with new authority artifacts rather than by rewriting historical verdict docs.
3. Round 3 现已用新的 authority-backed hypothesis 重开：当前 active line 已进一步收敛到 `layer0_query_distance`。只有当新的 round-3 authority artifact 真正改善 same-schema lane，才允许重开 family-level leadership 或 project-level acceptance 叙事。

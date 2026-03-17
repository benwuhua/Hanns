# Knowhere-RS Gap Analysis (Non-GPU)

Last updated: 2026-03-17
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

- ✅ 当前无活跃 P3 blocker；final rollup 已在 2026-03-17 收口为可接受状态。
- HNSW 当前 family verdict 为 `leading`，并以 near-equal-recall authority lane 作为项目级 leadership 证据来源。
- IVF-PQ 仍为 `no-go`，DiskANN 仍为 `constrained`；这两条结论作为实现边界持续有效。
- strict-ef same-schema lane（`ef=138`）继续保留为方法学/公平门工件，不直接承载项目级 leadership 判定。
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
  - strict-ef same-schema lane 与 near-equal-recall lane 的职责已分离并固化
  - strict-ef same-schema (`ef=138`) 当前为 native 小幅领先（`1.054x`）
  - 证据：`benchmark_results/baseline_p3_001_stop_go_verdict.json`、`benchmark_results/hnsw_fairness_gate.json`
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
  - `benchmark_results/final_core_path_classification.json` 当前为 HNSW=`leading`、IVF-PQ=`no-go`、DiskANN=`constrained`
- ✅ `FINAL-PERFORMANCE-LEADERSHIP-PROOF`: 最终 performance-leadership criterion 已归档为满足
  - `benchmark_results/final_performance_leadership_proof.json` 当前为 `criterion_met=true`
  - HNSW leadership 证据绑定 near-equal-recall authority lane（Rust `0.9518 / 28479.544` vs native `0.9500 / 15918.091`）
- ✅ `HNSW-P3-002`: **leading**
  - 当前 family 级最终结论已归档到 `benchmark_results/hnsw_p3_002_final_verdict.json`
- ✅ `FINAL-PRODUCTION-ACCEPTANCE`: 项目级最终 verdict 已归档为 `accepted`
  - `benchmark_results/final_production_acceptance.json` 当前记录 `production_accepted=true`

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

Current completion state:

1. `benchmark_results/final_production_acceptance.json` 当前项目级结论为 `accepted on current remote x86 evidence`。
2. 项目当前无活跃 tracked feature；后续仅在新 authority 证据触发时重开。
3. 领导力与方法学工件均已拆分并锁定：strict-ef same-schema 继续用于公平门，near-equal-recall 用于项目级 leadership。

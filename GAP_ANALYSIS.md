# Knowhere-RS Gap Analysis (Non-GPU)

Last updated: 2026-03-09 13:03 UTC  
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

- ✅ `CORE-P0-001`: 远端 x86 SIMD 验证链已恢复可执行并取得新鲜证据。最新复核中，本地 `cargo test --lib -q`、远端 `cargo test --features simd simd::tests -- --nocapture`、远端 `cargo test --lib --features simd test_x86_simd_l2_reduction_matches_scalar_on_irregular_input -- --nocapture` 均已通过；`default+simd` 不再因 toolchain/脚本漂移阻断后续核心路径工作。
- ✅ `HNSW-P1-001`: HNSW 已完成首轮远端 before/after artifact 落地；当前证据显示 recall 基本持平（`0.217 -> 0.215`）但 qps 大幅提升（`~1621 -> ~19235`），由于 recall 仍低于可信阈值，这条结果已被诚实归档为 `recheck required / no-go`，不再作为当前活动 blocker。
- 🚧 `IVFPQ-P1-002`: IVF/PQ 需要从“接口存在”切到“实现真实性和热点路径可信”。IVF base 当前更像占位实现，IVF-PQ/ScaNN 则需要 focused 审计和 benchmark 证明；这是当前最高优先级的活跃核心实现任务。
- 🚧 `DISKANN-P1-003`: Rust DiskANN 当前仍是“简化 Vamana + 简化 PQ”边界，需先修距离路径并明确工程边界，避免误把它当原生 DiskANN 同级实现。

- ✅ `SEM-P3-001`: `DiskANN` / `AISAQ` / `HNSW` / `IVF` / `Sparse` / `ScaNN` 的 `GetVectorByIds` / `HasRawData` Phase-5 语义尾项已完成 focused 收敛；当前不再缺“入口存在但边界语义不可解释”的 semantic-fidelity blocker。
- ✅ `ABI-P3-002`: FFI metadata / additional-scalar 已从“最小稳定摘要”升级为逐索引可解释 contract；HNSW / IVF / ScaNN / Sparse 的 capability/semantics/unsupported_reason 已具备 focused FFI 回归覆盖，当前不再是活跃 P3 blocker。
- ✅ `PERSIST-P3-003`: persistence / `DeserializeFromFile` 语义矩阵已系统化，`file_save_load` / `memory_serialize` / `deserialize_from_file` 的 supported / constrained / unsupported 边界现已可审计且具备 focused regressions。
- ✅ `OBS-P3-005`: 最小 runtime governance contract 已收口到 `knowhere_get_index_meta`，统一暴露 `observability` / `trace_propagation` / `resource_contract` 三组字段；当前不再缺“缺少稳定 schema/透传入口/资源口径”的 P3 blocker。
- ✅ `PERF-P3-004`: native benchmark harness 缺口已关闭；远端 x86 现已可构建 `benchmark_float_qps`、执行 `--gtest_list_tests`，且输出字段与 Rust parser 保持兼容。
- 🚧 `PERF-P3-005`: 当前性能基线任务仍保留，但它不再压过核心实现修补；现阶段需先等待 `IVFPQ-P1-002` 给出 go/no-go 结论，或等待 HNSW recall gate 达到可信阈值，然后再进入 native-vs-rs 主证据轮次。

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

Parity/governance tail-closure is closed, but the project as a whole is not:

1. `PARITY-P2-001` 已完成：HNSW-PQ 的高级接口/持久化语义已被稳定约束。
2. `OPT-P2-004` 已完成：`Index factory/legality` 的状态在 queue/gap/audit 中一致，不再因历史残留被误判为活跃缺口。
3. 下一阶段不再是“补入口”，而是“先把 `DISKANN / HNSW / IVF / PQ` 的核心实现做实，再谈 semantic fidelity / production hardening / performance advantage 的高质量收口”。

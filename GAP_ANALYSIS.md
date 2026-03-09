# Knowhere-RS Gap Analysis (Non-GPU)

Last updated: 2026-03-09  
Scope: Non-GPU production parity against C++ knowhere

## 1. Baseline and Method

- Rust repo: `/Users/ryan/.openclaw/workspace-builder/knowhere-rs`
- C++ repo: `/Users/ryan/Code/vectorDB/knowhere`
- Source of truth for parity status: `docs/PARITY_AUDIT.md`
- Priority order: BUG > PARITY > OPT > BENCH

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

## P3 (Semantic Fidelity / Production Readiness / Performance Advantage)

- ✅ `SEM-P3-001`: `DiskANN` / `AISAQ` / `HNSW` / `IVF` / `Sparse` / `ScaNN` 的 `GetVectorByIds` / `HasRawData` Phase-5 语义尾项已完成 focused 收敛；当前不再缺“入口存在但边界语义不可解释”的 semantic-fidelity blocker。
- ✅ `ABI-P3-002`: FFI metadata / additional-scalar 已从“最小稳定摘要”升级为逐索引可解释 contract；HNSW / IVF / ScaNN / Sparse 的 capability/semantics/unsupported_reason 已具备 focused FFI 回归覆盖，当前不再是活跃 P3 blocker。
- ✅ `PERSIST-P3-003`: persistence / `DeserializeFromFile` 语义矩阵已系统化，`file_save_load` / `memory_serialize` / `deserialize_from_file` 的 supported / constrained / unsupported 边界现已可审计且具备 focused regressions。
- 🚧 `OBS-P3-005`: 当前已有基础 tracing，但生产级 metrics / trace 透传 / 资源估算 contract 仍缺位；代码中现存的是零散 `tracing::*` 调用、benchmark memory estimator 与 legality `mmap_supported`，尚未收敛为统一 runtime governance contract。
- 🚧 `PERF-P3-004`: 项目目标已明确升级为“生产级平替 + 绝对性能优势”；仍需建立 native-vs-rs 的 recall-gated 优势基线。

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
3. 下一阶段不再是“补入口”，而是“semantic fidelity + production hardening + performance advantage”。

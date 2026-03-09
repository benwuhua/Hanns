# Knowhere-RS Gap Analysis (Non-GPU)

Last updated: 2026-03-08  
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

- 🚧 `SEM-P3-001`: `DiskANN` / `AISAQ` 已完成 metric-gated raw-data 语义收敛，但 `HNSW` / `IVF` / `Sparse` / `ScaNN` 的 `GetVectorByIds` / `HasRawData` 仍缺少 missing-id、empty-index、lossy-index、`has_raw_data=false` 的系统化语义矩阵。
- 🚧 `ABI-P3-002`: FFI metadata / additional-scalar 目前仍偏“最小稳定摘要”，需要向逐模块真实语义提升。
- 🚧 `PERSIST-P3-003`: persistence / `DeserializeFromFile` 语义矩阵尚未系统化，不足以直接宣称生产级替代完成。
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

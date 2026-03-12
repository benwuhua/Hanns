# Knowhere-RS Gap Analysis (Non-GPU)

Last updated: 2026-03-11 15:38 UTC
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
- ✅ `IVFPQ-P3-003`: **no-go**（recall@10≈0.442）
- ✅ `DISKANN-P3-004`: **no-go for parity**（`DiskAnnIndex` 是简化 Vamana + placeholder PQ；`PQFlashIndex` 是受限 AISAQ skeleton；默认 benchmark lane 与 compare lane 现已都显式披露并执行这一边界）
- ✅ `HNSW-P3-002`: **functional-but-not-leading**
  - layer-0 语义差异、same-schema HDF5 refresh、以及 HNSW FFI / persistence contract 均已 authority 收口
  - 当前 family 级最终结论已归档到 `benchmark_results/hnsw_p3_002_final_verdict.json`：Rust HNSW 具备可信 recall 与生产契约，但在当前 trusted same-schema lane 上 native 仍约快 `14.8x`，因此继续阻止 leadership claim
- 🚧 `IVFPQ-P3-003`: benchmark chain 已收口为可回放 no-go 证据
  - 已用 `ivfpq-hot-path-audit` 把真实 hot path 和 placeholder scaffold 分离
  - coarse/ADC regressions 与 remote benchmark refresh 均已完成；当前结论是不具备 leadership claim，而不是缺少 benchmark 事实
- [ ] `PROD-P3-005`: 最终生产验收门（只在前 4 个任务收口后复核）

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
3. 下一阶段不再是“补入口”，而是按大任务稳定推进：
   - 先建立可信基线
   - 再分别验证 `HNSW`、`IVF-PQ`
   - 再诚实收口 `DiskANN/PQ`
   - 最后统一过生产验收门。

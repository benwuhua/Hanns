# Builder 任务队列
> 最后更新: 2026-03-13 | 只保留当前大任务面板。历史任务已迁移到 `docs/TASK_QUEUE_ARCHIVE.md`。

## 当前大任务面板

- [ ] **HNSW-REOPEN-001**: 重开 HNSW 核心算法攻关线
  - 当前子阶段: `round8_parallel_build_graph_quality_baseline_frozen`
  - 当前结论: round 4 的 layer-0 parity cut 把 Rust same-schema HNSW 推到 `819.471` qps，但仍未改写历史 `functional-but-not-leading` 证据带；round 5 的 dispatch-cache rerun 与 stability gate 又表明后续比例变化仍受 native 波动影响，round 6/7 只完成了 prefetch 与 flat-graph 搜索路径审计，没有新的 authority verdict move。现在 `benchmark_results/hnsw_reopen_round8_baseline.json` 已经把这些 search-side 线索冻结成新的起点，下一条显式 hypothesis 转向并行构建图质量：当前 bulk-build 路径仍缺少 upper-layer greedy descent，并且 upper-layer overflow 仍在用 `truncate_to_best`，这比现有 serial/native 插入语义更弱。
  - 当前证据: `benchmark_results/hnsw_reopen_round8_baseline.json` + `benchmark_results/hnsw_reopen_round5_stability_gate.json` + `benchmark_results/hnsw_reopen_layer0_flat_graph_audit_round7.json`
  - 下一步: `hnsw-parallel-build-graph-audit-round8`。先把 bulk-build upper-layer descent / overflow shrink 差异锁成 round-8 audit artifact，再进入真正的 graph-quality rework 与 authority rerun。
  - 范围约束: 当前 reopen line 聚焦 `src/faiss/hnsw.rs` 的 bulk-build graph-quality parity；搜索侧 batch-4 dispatch caching 与 AVX2/FMA specialization 明确延后，不与首个 graph-quality rerun 混做归因。IVF-PQ、DiskANN、以及项目级 final acceptance 继续保持 archived state

- [x] **BASELINE-P3-001**: 建立可信的 native-vs-rs recall-gated 基线
  - 子阶段: `stop_go_verdict_formed` ✅
  - 结论:
    - 修复了方法论 bug：此前 `recall_at_10` 字段实际计算的是 recall@100（`--top-k 100` 传给了 recall k），已为 binary 增加独立 `--recall-at` 参数
    - 修复后 recall@10 测量确认：Rust HNSW 需要 **ef=2000** 才能达到当前 trusted native **ef=139** 同等的 recall@10≈0.9505
    - 差距量化：QPS 726 vs 15144.811（**20.9x 差距**）at recall@10≈0.95
    - 图结构存在（ef=5000 可达 recall@10=0.99），确认为**建图质量差距**，非搜索路径 bug
  - 证据: `benchmark_results/baseline_p3_001_stop_go_verdict.json`

- [x] **HNSW-P3-002**: HNSW 进入性能与生产契约收尾
  - 当前子阶段: `final_classification_archived` ✅
  - 当前状态: layer-0 邻居回填修复、authority HDF5 refresh、以及 HNSW FFI / persistence contract 已完成；family 级最终结论已归档为 `functional-but-not-leading`
  - 当前工作单: `memory/CURRENT_WORK_ORDER.json`
  - 完成标准: 基于当前 authority benchmark truth set 形成诚实的 leadership-or-no-go verdict
  - 当前结论: Rust HNSW 已关闭 recall 与生产契约缺口，但当前 trusted same-schema HDF5 lane 上 native 仍约快 `14.8x`；因此这条 archived verdict 现在只作为 HNSW reopen line 的历史基线，而不是永久停止信号

- [x] **IVFPQ-P3-003**: IVF-PQ family 最终 verdict 已归档
  - 当前子阶段: `final_classification_archived` ✅
  - 当前结论: `src/faiss/ivfpq.rs` 仍是 residual-PQ hot path，`src/faiss/ivf.rs` 仍是 coarse-assignment scaffold；现有 authority artifacts 一致表明 recall 未达 `0.8` gate，因此 family 级最终分类为 `no-go`
  - 证据: `benchmark_results/ivfpq_p3_003_final_verdict.json`，以及 `tests/bench_ivf_pq_perf.rs` / `tests/bench_recall_gated_baseline.rs` / `tests/bench_cross_dataset_sampling.rs` 的默认 lane regressions
  - 下一步: 无；除非 future authority evidence clears the recall gate, otherwise do not reopen IVF-PQ leadership or production-candidate claims

- [x] **DISKANN-P3-004**: Rust DiskANN family 最终 verdict 已归档
  - 当前子阶段: `family_final_classification_archived` ✅
  - 当前结论: `benchmark_results/diskann_p3_004_benchmark_gate.json` 继续把 benchmark lane 固化为 `no_go_for_native_comparable_benchmark`，而新的 `benchmark_results/diskann_p3_004_final_verdict.json` 把 family-level final classification 固化为 `constrained`。也就是说，`src/faiss/diskann.rs` 与 `src/faiss/diskann_aisaq.rs` 仍是功能可用但简化的 Vamana/AISAQ 实现，不允许进入 native-comparable benchmark 或 leadership claim。
  - 证据: `benchmark_results/diskann_p3_004_final_verdict.json`，`benchmark_results/diskann_p3_004_benchmark_gate.json`，`benchmark_results/cross_dataset_sampling.json`，`src/faiss/diskann.rs` / `src/faiss/diskann_aisaq.rs` 的 scope audit，`tests/bench_diskann_1m.rs` / `tests/bench_compare.rs` 的 final-verdict regressions
  - 下一步: 无；family-level DiskANN classification 已关闭，转入跨 family 的 `final-core-path-classification`

- [x] **PROD-P3-005**: 最终生产验收门 verdict 已归档
  - 当前子阶段: `final_acceptance_verdict_archived` ✅
  - 当前结论: production engineering gates 已全部收口，但项目最终 verdict 明确为 `not accepted`；当前 authority evidence 仍不支持 non-GPU production replacement claim
  - 证据: `benchmark_results/final_production_acceptance.json`、`benchmark_results/final_core_path_classification.json`、`benchmark_results/final_performance_leadership_proof.json`
  - 下一步: 无；除非新的 authority artifact 实质改变 leadership 或 core-path verdict chain，否则不要重开正向 final acceptance claim

## 粒度约定

- `task_id` 只表示大任务，不表示 blocker、脚本问题或证据缺口
- 具体 blocker / 子阶段 / 当前动作统一进入：
  - `memory/PLAN_RESULT.json`
  - `memory/EXEC_RESULT.json`
  - `memory/CURRENT_WORK_ORDER.json`
- 每轮默认继续当前工作单；只有以下情况才切换大任务：
  - 当前大任务完成
  - 当前大任务被证据明确判成 no-go
  - 项目阶段切换

# Builder 任务队列
> 最后更新: 2026-03-10 20:00 UTC | 只保留当前大任务面板。历史任务已迁移到 `docs/TASK_QUEUE_ARCHIVE.md`。

## 当前五个大任务

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
  - 当前结论: Rust HNSW 已关闭 recall 与生产契约缺口，但当前 trusted same-schema HDF5 lane 上 native 仍约快 `14.8x`；因此 HNSW 保持可用但不得宣称 performance leadership

- [ ] **IVFPQ-P3-003**: 审计并锁定真实 IVF-PQ hot path
  - 当前子阶段: `no_go_evidence_archived`
  - 当前结论: `src/faiss/ivfpq.rs` 是 residual-PQ hot path；`src/faiss/ivf.rs` 只是 coarse-assignment scaffold，remote benchmark chain 已完成，但结果仍不足以支撑 parity / leadership claim
  - 当前工作单: `memory/CURRENT_WORK_ORDER.json`
  - 下一步: 仅在后续需要补 production contract 或 stop-go 归档细节时再回到 IVF-PQ family

- [x] **DISKANN-P3-004**: Rust DiskANN 边界已明确
  - 结论: `src/faiss/diskann.rs` 仍是简化 Vamana + placeholder PQCode（均值量化）；`src/faiss/diskann_aisaq.rs` 暴露了真实 flash-layout / beam-search / page-cache skeleton，但仍是简化 AISAQ 路径，不具备原生 SSD 管道能力，**no-go** for C++ DiskANN 性能对比
  - 证据: `docs/GAP_ANALYSIS.md`，`src/faiss/diskann.rs` / `src/faiss/diskann_aisaq.rs` 的 scope audit，`tests/bench_diskann_1m.rs` 的 scope-disclosure regression，`tests/bench_compare.rs` 的 compare-lane exclusion regression

- [ ] **PROD-P3-005**: 最终生产验收门
  - 范围: semantic fidelity、persistence / deserialize-from-file、FFI metadata / additional scalar、minimum observability / runtime governance
  - 说明: 只在前 4 个大任务收口后统一复核

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

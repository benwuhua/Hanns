# Builder 任务队列
> 最后更新: 2026-03-18 | 只保留当前大任务面板。历史任务已迁移到 `docs/TASK_QUEUE_ARCHIVE.md`。
> 详细 gap analysis: `docs/superpowers/specs/2026-03-18-diskann-aisaq-gap-analysis.md`

## 当前大任务面板

> 优先级调整 2026-03-18: HNSW/DiskANN/PQ/SQ/IVF/ScaNN 整体上移；AISAQ 降级。

### Phase 3: 多 Index 能力验证 + 修复 (2026-03-18 开启)

#### P0 — 验证/诊断（立即）

- [ ] **HNSW-VERIFY-001** [P0]: 独立重跑 near-equal-recall benchmark 验证 1.789x 结论
  - 在 x86 上同时跑 strict-ef lane (ef=138) + near-equal-recall lane (Rust at recall≈0.95)
  - 对比 native ef=138 (15,918 QPS, recall=0.95)
  - Codex 独立完成的结论不能完全信任，必须 Claude 参与验证

- [ ] **IVF-FLAT-001** [P0]: nprobe 参数扫描 → 确认 IVF-Flat 能过 0.95 recall gate
  - 当前 nprobe=8 → recall=0.535，太低
  - 预期 nprobe=32-64 可达 0.95+
  - 得到 x86 authority QPS

- [x] **DISKANN-RECALL-001** [P0]: ✅ 诊断完成
  - 根因: `add()` 不构建图边；只有 `train()` 中的向量有 Vamana 图连接
  - cross_dataset recall=0.009 来自 train(空/centroid)+add(4K) → 所有节点 degree=0
  - 真实 dim 影响: 单调递减 (dim=64: 0.91, dim=128: 0.82)，非 dim=64 特有
  - 修复路径: add() 调用 insert_point() 接入图，或 API 文档明确 train(完整数据集)
  - 证据: `examples/diskann_dim_diag.rs`

#### P1 — 填空白基线

- [x] **IVF-SQ8-001** [P1]: ✅ 完成 → 转 no-go
  - recall@full_scan=0.174 (plateau)，根因: quantizer.train(vectors) 应为 quantizer.train(&residuals)
  - 负数 residual 全被 clamp 到 0 → 距离计算错误 → recall 受限于量化上限
  - 证据: `examples/ivf_sq8_sweep.rs`
  - 修复: 计算完 k-means 后再用所有 residuals 训练 quantizer (P2 fix)
- [x] **IVF-OPQ-001** [P1]: ✅ done → no-go
  - recall@full_scan=0.167 (plateau), same quantization accuracy issue as IVF-SQ8
  - Script: `examples/ivf_opq_sweep.rs`
- [x] **SCANN-001** [P1]: ✅ done → no-go
  - max recall@10: 0.699 at reorder_k=160 (below 0.95 gate), QPS=154
  - recall increases with reorder_k but far from gate at practical settings
  - Script: `examples/scann_sweep.rs`
- [x] **IVF-RABITQ-001** [P1]: ✅ done → no-go
  - no_refine recall: 0.260 (constant across all nprobe, 1-bit quantization lossy)
  - with_refine(dataview, k=40) recall: 0.552 (constant, still below 0.95 gate)
  - Script: `examples/ivf_rabitq_sweep.rs`

#### P2 — 修复已知问题

- [ ] **IVF-PQ-FIX-001** [P2]: IVF-PQ recall 0.47 根因分析 + 修复（目前 no-go）
- [x] **IVF-SQ8-FIX-001** [P2]: ✅ 修复完成
  - 修复: train() 改为先 train_ivf() 获得 centroids，再用 residuals 训练 quantizer
  - 结果: recall@full_scan 从 0.174 → 0.982（100K random, nprobe=256, Mac）
  - 达到 recall gate (≥0.95 at nprobe=256，random worst-case)
  - 需要 SIFT-1M 测试确认 authority QPS
- [ ] **IVF-RABITQ-FIX-001** [P2]: IVF-RaBitQ recall ceiling — investigate larger refine_k (100x) or better quantization
- [ ] **SCANN-FIX-001** [P2]: ScaNN recall ceiling (0.699 max at 100K) — investigate larger num_centroids/reorder_k or algorithm fix
- [ ] **HNSW-IMP-001** [P2]: 若 strict-ef 仍落后 native，找真正的优化方向

### AISAQ Phase 2: 能力补全 + 生产就绪 (2026-03-18 开启, 降级为 P2+)

- [ ] **AISAQ-CAP-001** [P0]: 真正的 exact rerank stage
  - 现状: `rearrange_candidates` 用 ADC 非 exact distance
  - 目标: beam search 后取 top-N 候选，用原始 float 向量重算精确距离再排序
  - 参考: `diskann-disk/src/search/provider/disk_provider.rs` `post_process()`
  - 完成标准: recall@10 可测提升 + authority A/B

- [ ] **AISAQ-CAP-002** [P1]: External ID / Tag 系统
  - `i64` external_id ↔ internal `u32` row_id 双向映射
  - Milvus 集成前必须: 搜索结果返回 external IDs
  - 设计: `BTreeMap<i64, u32>` + `Vec<i64>` + 序列化支持

- [ ] **AISAQ-CAP-003** [P1]: On-disk persistence
  - 现状: materialize_storage() 是内存加速，重启即丢
  - 目标: CSR graph + PQ codebooks + raw vectors → mmap 文件；save()/load()
  - 参考: native knowhere FileManager + Rust DiskANN DiskIndexWriter

- [ ] **AISAQ-CAP-004** [P1]: RangeSearch
  - 参考: native `DiskANNIndexNode::RangeSearch` with range_filter

- [ ] **AISAQ-CAP-005** [P2]: Incremental insert + lazy delete + consolidation
  - 参考: Rust DiskANN `add()` / `consolidate_vector()`

- [ ] **AISAQ-CAP-006** [P2]: Multi-entry-point medoid seeding

- [ ] **AISAQ-CAP-007** [P2]: x86 SIMD audit (AVX2/AVX-512 验证)
  - 当前 x86 比 Mac 慢 2.25x；部分可能是 SIMD 未激活

- [ ] **AISAQ-CAP-008** [P2]: Async IO (io_uring) 冷 disk 路径
  - 现状: 冷盘 ~330 QPS；目标 async batch reads 达到 5K+ QPS

- [ ] **IVFPQ-FIX-001** [P3]: IVF-PQ recall < 0.8 根因分析+修复

### 本轮已完成 (2026-03-18)

- [x] **HNSW-REOPEN-001**: HNSW 重开线已归档
  - 当前子阶段: `closed_after_opt56_authority_and_final_rollup` ✅
  - 当前结论: 通过 `opt53 + opt56` 的 authority 证据，HNSW family verdict 已刷新为 `leading`，并进入项目级最终链路。
  - 证据: `benchmark_results/hnsw_p3_002_final_verdict.json`、`benchmark_results/final_performance_leadership_proof.json`
  - 下一步: 无；除非未来出现新的 authority 回归证据需要重新开线。

- [x] **BASELINE-P3-001**: 建立可信的 native-vs-rs recall-gated 基线
  - 子阶段: `stop_go_verdict_formed` ✅
  - 结论:
    - strict-ef same-schema 口径已固定为方法学对齐参考，不再直接承载项目级 leadership 结论
    - 当前 strict-ef lane (`ef=138`) 为 native 小幅领先（`1.054x`）
  - 证据: `benchmark_results/baseline_p3_001_stop_go_verdict.json`

- [x] **HNSW-P3-002**: HNSW 进入性能与生产契约收尾
  - 当前子阶段: `final_classification_archived` ✅
  - 当前状态: layer-0 与搜索热路径优化、authority 刷新、以及 HNSW FFI / persistence contract 已完成；family 级最终结论已归档为 `leading`
  - 当前工作单: `memory/CURRENT_WORK_ORDER.json`
  - 完成标准: 基于当前 authority benchmark truth set 形成诚实的 leadership-or-no-go verdict
  - 当前结论: near-equal-recall authority lane 上 Rust 已形成可信 leadership（`Rust/Native=1.789x`）

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
  - 当前结论: production engineering gates 已全部收口，项目最终 verdict 为 `accepted`
  - 证据: `benchmark_results/final_production_acceptance.json`、`benchmark_results/final_core_path_classification.json`、`benchmark_results/final_performance_leadership_proof.json`
  - 下一步: 无；除非新的 authority artifact 实质改变 leadership 或 core-path verdict chain。

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

# Builder 任务队列
> 最后更新: 2026-03-05 19:28 | 优先级: BUG > PARITY > OPT > BENCH

## 待办 (TODO)

### P0 (紧急)
- [x] **BUG-P0-001**: 修复 `mini_batch_kmeans` SIMD 长度不匹配导致的测试失败 (2026-03-05)
  - 失败用例: `clustering::mini_batch_kmeans::tests::test_mini_batch_kmeans_large_dataset`
  - 现象: `src/simd.rs` 中 `l2_distance`/`l2_distance_sq` 长度断言触发
  - 修复: 在 `init_centroids`/`find_nearest_centroid`/`process_batch` 中修正切片长度为 dim
  - 验收: 所有 mini_batch_kmeans 测试通过 (7/7)
- [x] **BUG-P0-002**: 修复 `diskann_complete` 批量 add 路径维度切片错误 (2026-03-05)
  - 失败用例: `faiss::diskann_complete::tests::test_diskann_add_batch`
  - 现象: `src/simd.rs` 长度断言触发（8 vs 16）
  - 修复: 在 `add_batch` 中修正切片长度为 dim
  - 验收: 所有 diskann_complete 测试通过 (5/5)
- [ ] **BUG-P0-003**: 修复 `ivf_sq_cc` 系列并发/检索路径的维度不一致
  - 失败用例:
    - `faiss::ivf_sq_cc::tests::test_ivf_sq_cc_concurrent_add`
    - `faiss::ivf_sq_cc::tests::test_ivf_sq_cc_concurrent_mixed`
    - `faiss::ivf_sq_cc::tests::test_ivf_sq_cc_get_vectors`
    - `faiss::ivf_sq_cc::tests::test_ivf_sq_cc_concurrent_search`
    - `faiss::ivf_sq_cc::tests::test_ivf_sq_cc_train_add_search`
  - 现象: `src/simd.rs` 长度断言触发（4 vs 8 等）
  - 验收: `ivf_sq_cc` 相关测试全部通过，`cargo test --lib` 不新增失败
- [ ] **PARITY-P0-001**: 统一核心索引契约行为（Build/Train/Add/Search/RangeSearch/AnnIterator/GetVectorByIds/Serialize/Deserialize）
  - 验收: 非 GPU 核心索引在契约层行为一致，`docs/PARITY_AUDIT.md` 的相关项变更为 Done。
- [ ] **PARITY-P0-002**: 修复 FFI 能力声明与运行时不一致问题
  - 验收: `src/ffi.rs` 索引能力矩阵与实际构造/调用路径一致；无“声明支持但运行时 NotImplemented”错位。

### P1 (重要)
- [ ] **PARITY-P1-001**: HNSW 高级路径对齐（range/iterator/get-by-id/serialize 语义）
  - 验收: 对齐 C++ 行为并补齐对应测试。
- [ ] **PARITY-P1-002**: IVF 系列参数与边界行为对齐（含 IVFPQ/IVFSQ/RaBitQ）
  - 验收: 参数校验、错误路径和搜索结果语义与审计表一致。
- [ ] **PARITY-P1-003**: DiskANN/AISAQ 生命周期与参数语义对齐
  - 验收: 明确并实现与 C++一致或有文档化差异的行为；补充回归测试。
- [ ] **PARITY-P1-004**: 建立 index × datatype × metric 合法性统一校验层
  - 验收: 非法组合在入口层被阻断并返回一致错误码。

### P2 (优化)
- [ ] **OPT-P2-001**: 统一 benchmark 报告模板并强制 ground truth 来源字段
  - 验收: benchmark 报告自动包含 ground truth 来源、R@10 和可信度标记。
- [ ] **OPT-P2-002**: 扩展回归测试覆盖到 300+ 稳定用例
  - 验收: 测试数量达标且无新增 flaky 模块。
- [ ] **BENCH-P2-001**: 建立 recall-gated 性能对标流水（Rust vs C++）
  - 验收: 报告包含差距、可信度和重验标记。

## 归档
- [x] **DOCS-BASELINE-001**: 重建 GAP/ROADMAP/TASK_QUEUE/PARITY_AUDIT 文档基线（2026-03-05）

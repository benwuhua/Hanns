# Builder 任务队列
> 最后更新: 2026-03-06 03:35 | 优先级: BUG > PARITY > OPT > BENCH

## 待办 (TODO)

### P0 (紧急)
- [x] **BUG-P0-001**: 修复 `mini_batch_kmeans` SIMD 长度不匹配导致的测试失败 (2026-03-05)
- [x] **BUG-P0-002**: 修复 `diskann_complete` 批量 add 路径维度切片错误 (2026-03-05)
- [x] **BUG-P0-003**: 修复 `ivf_sq_cc` 系列并发/检索路径的维度不一致 (2026-03-05)
- [x] **PARITY-P0-001**: 统一核心索引契约行为（Build/Train/Add/Search/RangeSearch/AnnIterator/GetVectorByIds/Serialize/Deserialize）
  - 进展: ✅ 核心契约一致性验证完成 (2026-03-06 03:35)
  - 验收: ✅ 所有非 GPU 索引在契约层行为一致，Index trait 默认 Unsupported 实现，FFI 层 19 处 NotImplemented 返回，`docs/PARITY_AUDIT.md` Core contract 状态变更为 Done。
- [x] **PARITY-P0-002**: 修复 FFI 能力声明与运行时不一致问题 (2026-03-06)
  - 进展: ✅ 添加 FFI AnnIterator 接口 (`knowhere_create_ann_iterator`/`knowhere_ann_iterator_next`/`knowhere_free_ann_iterator`)
  - 验收: `src/ffi.rs` 索引能力矩阵与实际构造/调用路径一致；无"声明支持但运行时 NotImplemented"错位。

### P1 (重要)
- [x] **PARITY-P1-000**: 为核心索引实现 AnnIterator 接口（HNSW/IVF/Flat）
  - 实现状态:
    - ✅ HNSW: `src/faiss/hnsw.rs:2432-2492` - HnswAnnIterator
    - ✅ ScaNN: `src/faiss/scann.rs:999-1034` - ScannAnnIterator
    - ✅ HNSW-PQ: `src/faiss/hnsw_pq.rs:719-757` - HnswPqAnnIterator
    - ✅ DiskANN: 已有实现 `src/faiss/diskann.rs:961-1000` - DiskAnnIterator
  - 验收标准: 至少 3 个核心索引实现 AnnIterator ✅ (实际 4 个)
- [ ] **PARITY-P1-001**: HNSW 高级路径对齐（range/iterator/get-by-id/serialize 语义）
  - 进展: ✅ get_vector_by_ids 实现 (hnsw.rs:2402-2431)
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
- [x] **DOCS-BASELINE-002**: 创建 FFI 能力矩阵文档 `docs/FFI_CAPABILITY_MATRIX.md` (2026-03-05)
- [x] **PARITY-P0-003**: 添加 AnnIterator trait 定义到 `src/index.rs` (2026-03-05)
- [x] **DOCS-BASELINE-001**: 重建 GAP/ROADMAP/TASK_QUEUE/PARITY_AUDIT 文档基线（2026-03-05）

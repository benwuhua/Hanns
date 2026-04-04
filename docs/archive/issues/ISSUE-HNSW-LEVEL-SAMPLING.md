# ISSUE-HNSW-LEVEL-SAMPLING
**日期**: 2026-03-24 | **严重程度**: P1
## 问题
`src/faiss/hnsw.rs` 的层级分配公式使用了固定参考常数而不是 runtime `M`：
- `REFERENCE_M_FOR_LEVEL = 16`（`src/faiss/hnsw.rs:56`）
- `level_multiplier` 默认 `1.0 / ln(REFERENCE_M_FOR_LEVEL)`（`src/faiss/hnsw.rs:1361-1364`）
- `random_level` 为 `(-ln(U) * level_multiplier)`（`src/faiss/hnsw.rs:1446-1451`）

这会让不同 `M` 下的层分布保持相同，而不是随 `M` 变化。

## Native 对比
hnswlib 标准实现是基于当前 `M` 计算层级分布：
- `mult_ = 1 / log(1.0 * M_)`（`.../thirdparty/hnswlib/hnswlib/hnswalg.h:129`）
- `getRandomLevel(reverse_size)` -> `-log(U) * reverse_size`（`.../hnswalg.h:218-221`）
- 新点插入用 `getRandomLevel(mult_)`（`.../hnswalg.h:1052`）

## 影响
当用户显式调整 `M`（例如从 16 到 32/64）时，Rust 实现的层数分布与 native/hnswlib 行为不一致，导致图拓扑统计特征偏移，可能带来 recall/延迟曲线偏差，难以直接对齐 native 基准。

## 建议方向
将默认层级分布恢复为 `ml = 1/ln(M)`（保持可配置 override），并在对齐模式下提供“strict hnswlib level policy”开关，避免历史行为回归风险。

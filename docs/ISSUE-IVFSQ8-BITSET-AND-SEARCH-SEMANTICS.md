# ISSUE-IVFSQ8-BITSET-AND-SEARCH-SEMANTICS
**日期**: 2026-03-24 | **严重程度**: P1
## 问题
`src/faiss/ivf_sq8.rs` 的 bitset 过滤语义与 native IVF 搜索语义不一致：
1. bitset 过滤发生在“完成 top-k 检索之后”的结果后处理，而非搜索过程内过滤（`src/faiss/ivf_sq8.rs:1004-1018`）。
2. bitset 判定使用 `id as usize` 作为位图索引（`src/faiss/ivf_sq8.rs:1012-1014`），当使用自定义外部 ID 时会错位（bitset 在系统里通常按内部 row/idx 对齐）。

## Native 对比
native IVF 路径在搜索内直接把 bitset 传给底层搜索 API（thread-safe search/range）：
- KNN: `search_*_thread_safe(..., nprobe, ..., bitset)`（`/Users/ryan/Code/knowhere/src/index/ivf/ivf.cc:390`, `398-403`）
- Range: `range_search_*_thread_safe(..., bitset)`（`ivf.cc:465-471`）

即 native 在候选生成阶段就过滤，而不是返回后再删。

## 影响
- 后过滤会导致 top-k 被过滤后数量不足，且不会补齐正确候选，召回和结果稳定性劣化。
- 自定义 ID 场景下 bitset 错位会误过滤/漏过滤，行为不可预测。

## 建议方向
- 将 bitset 过滤前移到 cluster scan 阶段，按内部向量索引（row id）过滤。
- 在 API 层明确区分 external id 与 internal idx，禁止 `id as idx` 假设。
- 增加自定义 ID + bitset 的 parity 回归测试，对齐 native 行为。

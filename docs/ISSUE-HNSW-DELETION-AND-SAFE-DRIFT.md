# ISSUE-HNSW-DELETION-AND-SAFE-DRIFT
**日期**: 2026-03-24 | **严重程度**: P2
## 问题
当前 HNSW 删除语义和备用实现都与 native/hnswlib 规范存在明显差距：
1. `src/faiss/hnsw.rs` 没有内建节点删除/恢复 API（无内部 deleted mask、无图内懒删除重用），仅通过查询 bitset 做外部过滤（注释中称“e.g., soft-deleted vectors”）：`src/faiss/hnsw.rs:3730-3734`。
2. `src/faiss/hnsw_safe.rs`（虽标记 deprecated）并非 HNSW 图搜索实现：`search()`/`range_search()` 为全量暴力扫描（`src/faiss/hnsw_safe.rs:106-137`, `139-162`），`save/load` 也只是占位（`179-185`）。

## Native 对比
native knowhere 使用 `hnswlib::HierarchicalNSW` 作为唯一实现核心（`/Users/ryan/Code/knowhere/src/index/hnsw/hnsw.cc:59-60`），查询走 `searchKnn/searchRange`（`hnsw.cc:129`, `203`）。

hnswlib 规范通常提供懒删除语义（mark-delete family）；本仓当前 thirdparty 分支未暴露该接口，但 native 路径至少保持单一、图搜索一致的实现，不存在“安全版=暴力检索”这种行为分叉。

## 影响
- 删除能力依赖上层 bitset 约束，缺少索引内生命周期管理，长期运行下可能带来图质量/内存回收策略不明确。
- `hnsw_safe.rs` 若被误用，性能和行为会与 HNSW 预期严重偏离，且持久化不可用，增加维护风险。

## 建议方向
- 明确官方删除语义：要么实现索引内 lazy deletion，要么在 API 文档中声明“仅外部 bitset 删除”。
- 将 `hnsw_safe.rs` 从可选实现收敛为实验/测试模块，避免被生产路径误接入；或补齐为真实 HNSW 语义再保留。

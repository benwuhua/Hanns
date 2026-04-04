# ISSUE-HNSW-EF-CONTRACT
**日期**: 2026-03-24 | **严重程度**: P1
## 问题
Rust HNSW 的 `ef` 语义与 native 不一致，存在两处偏差：
1. 查询时把 `SearchRequest.nprobe` 作为 `requested_ef_search` 参与计算（`src/faiss/hnsw.rs:3636-3640`, `3703-3707`, `3763-3767`），再由 `effective_hnsw_ef_search` 取 `max(base_ef, nprobe, adaptive_floor)`（`src/api/index.rs:520-530`）。
2. 构建时 `ef_construction` 仅做 `max(1)`（`src/faiss/hnsw.rs:1355-1356`），没有像 hnswlib 那样强制 `ef_construction >= M`。

## Native 对比
native knowhere 直接把搜索 `ef` 传入 hnswlib：
- `SearchParam{(size_t)hnsw_cfg.ef.value(), ...}`（`/Users/ryan/Code/knowhere/src/index/hnsw/hnsw.cc:120`）
- `hnsw_config` 还要求 `ef >= k`（`/Users/ryan/Code/knowhere/src/index/hnsw/hnsw_config.h:55-63`）

hnswlib 构造时明确做了：
- `ef_construction_ = std::max(ef_construction, M_)`（`.../thirdparty/hnswlib/hnswlib/hnswalg.h:95`）

## 影响
参数语义会和 native 基准错位：同样“ef=xxx”在 Rust 侧可能被 `nprobe`/adaptive floor 放大或改写；小 `ef_construction` 场景下图质量行为也与 hnswlib 预期不同，导致 recall/QPS 对比不稳定。

## 建议方向
统一 HNSW 参数语义：
- 查询参数引入明确的 `ef` 字段（或 `hnsw_ef_search_override`），不要复用 `nprobe`。
- 构建阶段对齐 `ef_construction = max(ef_construction, M)`。
- 保留 adaptive 策略时，做成显式可开关策略而不是默认隐式叠加。

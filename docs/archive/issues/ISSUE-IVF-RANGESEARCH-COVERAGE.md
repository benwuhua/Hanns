# ISSUE-IVF-RANGESEARCH-COVERAGE
**日期**: 2026-03-24 | **严重程度**: P2
## 问题
Rust IVF 家族当前仅覆盖 KNN 主路径，range search parity 不完整：
- `ivf.rs` 仅有 `search/search_with_bitset`（`src/faiss/ivf.rs:81-146`）。
- `ivfpq.rs` 仅实现 `search/search_parallel`（`src/faiss/ivfpq.rs:466-751`）。
- `ivf_sq8.rs` 仅实现 `search/search_parallel`（`src/faiss/ivf_sq8.rs:327-413`, `643-712`）。

## Native 对比
native knowhere IVF 对各 IVF 类型统一提供 RangeSearch，并在底层调用 `range_search_*_thread_safe`：
- `IvfIndexNode<T>::RangeSearch(...)` 主流程（`/Users/ryan/Code/knowhere/src/index/ivf/ivf.cc:422-497`）
- 具体调用 `range_search_thread_safe` / `range_search_without_codes_thread_safe`（`ivf.cc:465-471`）

## 影响
- Rust IVF 家族在 API 能力上低于 native，导致上层依赖 range 查询场景时无法等价替换。
- 无法对齐 native 的 radius/range_filter 语义与性能回归基线。

## 建议方向
- 为 IVF-SQ8/IVFPQ/IVF-flat parity 路径统一补齐 range search API。
- 复用 native 同步语义：`radius + range_filter + bitset`，并增加 cross-check 测试。
- 在文档中明确当前支持矩阵，避免功能误判。

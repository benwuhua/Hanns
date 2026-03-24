# ISSUE-FFI-SEARCH-CONTRACT-GAPS
**日期**: 2026-03-24 | **严重程度**: P1
## 问题
FFI 的搜索契约在不同 index 上存在“静默降级/未实现”路径：
- `search_with_bitset` 对“其他索引类型”回退到普通搜索（忽略 bitset）`src/ffi.rs:2288-2290`。
- `range_search` 仅对 Flat 实现，HNSW/ScaNN 返回 `NotImplemented`（`src/ffi.rs:1279-1303`，并有对应测试 `5609-5635`）。

## Native 对比（引用文件+行号）
- native `IndexNode` 抽象层把 `Search` 与 `RangeSearch` 都定义为标准能力，且都携带 `BitsetView`：
  - `Search(..., const BitsetView&)`：`/Users/ryan/Code/knowhere/include/knowhere/index_node.h:38-40`
  - `RangeSearch(..., const BitsetView&)`：`/Users/ryan/Code/knowhere/include/knowhere/index_node.h:41-43`
- native `Index` 包装同样走统一 `Search/RangeSearch` 配置校验流程：
  - `Search`：`/Users/ryan/Code/knowhere/include/knowhere/index.h:156-177`
  - `RangeSearch`：`/Users/ryan/Code/knowhere/include/knowhere/index.h:179-189`

## 影响
- 业务调用 `search_with_bitset` 时可能误以为过滤生效，实际被回退成无过滤搜索。
- `range_search` 能力在不同 index 类型上行为分裂，增加 API 语义不确定性。
- 对标 native 时，功能覆盖统计会被“接口存在但语义不一致”扭曲。

## 建议方向
- 禁止静默降级：bitset 不支持时返回明确错误码与错误信息。
- 为每个 index 在 meta 中显式声明 `search_with_bitset`/`range_search` 支持状态，并在 C API 暴露。
- 按 native 抽象契约补齐核心 index 的 range_search，或统一标记 unsupported（但不隐式回退）。

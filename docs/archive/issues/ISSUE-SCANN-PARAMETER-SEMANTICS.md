# ISSUE-SCANN-PARAMETER-SEMANTICS
**日期**: 2026-03-24 | **严重程度**: P2
## 问题
ScaNN 路径的参数语义与 FFI 公共配置不一致：
- FFI 配置包含 `metric_type`/`ef_search`（`src/ffi.rs:111-117`），但 ScaNN 创建仅消费 `num_partitions/num_centroids/reorder_k`（`src/ffi.rs:423-440`）。
- ScaNN实现固定 L2：`metric_type()` 直接返回 L2（`src/faiss/scann.rs:911-915`），FFI 也“假设 L2”（`src/ffi.rs:1458-1460`）。
- 搜索主流程仅 `coarse_search(..., reorder_k)` + rerank（`src/faiss/scann.rs:495-500`），没有 `ef_search` 等效控制项。

## Native 对比（引用文件+行号）
- native index type 常量未包含 ScaNN：`/Users/ryan/Code/knowhere/include/knowhere/comp/index_param.h:24-44`
- native 索引抽象是统一配置校验路径（`Search/RangeSearch`）：`/Users/ryan/Code/knowhere/include/knowhere/index.h:156-189`

## 影响
- 调用者传入 `metric_type=IP/Cosine` 或调整 `ef_search` 时，ScaNN 路径可能“参数被接受但未生效”。
- 容易造成 benchmark 配置误解，影响 QPS/recall 对比结论可信度。

## 建议方向
- 在 FFI 层为 ScaNN 做参数白名单校验：拒绝无效 `metric_type/ef_search`。
- meta 输出中增加“ScaNN 仅 L2 + reorder_k 主控”的能力声明。
- 若目标是 native parity，应先明确 ScaNN 在 native 体系中的定位（实验特性或非 parity 特性）。

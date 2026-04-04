# ISSUE-SPARSE-METRIC-AND-SSIZE-DRIFT
**日期**: 2026-03-24 | **严重程度**: P1
## 问题
Sparse 路径存在参数语义漂移：
- FFI 创建 `SparseWand/SparseWandCc` 时，除 IP 外所有 metric 都强制映射为 IP（`src/ffi.rs:867-870`, `895-898`）。
- `SparseWandIndexCC::new(metric_type, ssize)` 中 `ssize` 被显式忽略（`src/faiss/sparse_wand_cc.rs:48-50`）。

## Native 对比（引用文件+行号）
- native index type 常量未定义 Sparse/WAND 家族：`/Users/ryan/Code/knowhere/include/knowhere/comp/index_param.h:24-44`
- Rust sparse 注释引用了 native 路径 `src/index/sparse/sparse_inverted_index.h`（`src/faiss/sparse_inverted.rs:3`），但当前 native 仓库无对应实现入口（`src/index/` 下无 sparse 子目录）。

## 影响
- API 层参数“可传入”但不等价执行，调用者难以判断实际检索语义。
- `ssize` 形参无效会误导调优，造成性能/吞吐预期偏差。
- 在无 native 对照实现时，Sparse parity 评估缺乏基线，容易产生错误对标结论。

## 建议方向
- 对 Sparse 路径做严格参数校验：不支持的 metric 直接报错而非 silent remap。
- 若 `ssize` 暂不生效，移出公开配置或在 meta 中标记 no-op。
- 明确 Sparse 的对标状态（native parity / experimental），并在文档中单独声明。

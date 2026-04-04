# ISSUE-FFI-INDEX-TYPE-COVERAGE
**日期**: 2026-03-24 | **严重程度**: P1
## 问题
`src/ffi.rs` 的 `CIndexType` 暴露面与 native knowhere 的 index type 枚举不对齐：
- Rust FFI 暴露了 `Scann`、`SparseWand`、`SparseWandCc`（`src/ffi.rs:77`, `88`, `90`）。
- 但 native index type 常量集中在 `FLAT/IVF/HNSW/DISKANN/GPU_*`（`include/knowhere/comp/index_param.h:24-44`），不存在 ScaNN/Sparse 条目。
- 反向上，native 有 `INDEX_DISKANN` 与 `INDEX_RAFT_CAGRA`（`index_param.h:40,43`），而 Rust FFI `CIndexType` 中没有对应项（`src/ffi.rs:74-94`）。

## Native 对比（引用文件+行号）
- native index type 常量：`/Users/ryan/Code/knowhere/include/knowhere/comp/index_param.h:24-44`
- Rust FFI index type 列表：`src/ffi.rs:74-94`

## 影响
- 跨语言调用层无法形成稳定的“同名 index type = 同实现”契约。
- 上层若按 native 类型（例如 DISKANN/CAGRA）下发配置，会在 Rust FFI 侧无直接映射。
- benchmark 与线上配置迁移时易出现“名字存在但语义不一致/名字缺失”的集成风险。

## 建议方向
- 建立一份明确的 FFI type 对齐矩阵（native ↔ rust）。
- 对 native 已有但 Rust FFI 缺失的类型（至少 DISKANN/CAGRA）明确策略：补齐或显式声明 unsupported。
- 对 Rust 独有类型（ScaNN/Sparse）增加“非 native parity”标记，避免被误认为可与 native 一致对标。

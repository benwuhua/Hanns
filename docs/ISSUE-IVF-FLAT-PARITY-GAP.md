# ISSUE-IVF-FLAT-PARITY-GAP
**日期**: 2026-03-24 | **严重程度**: P1
## 问题
`src/faiss/ivf.rs` 当前是简化版 IVF 容器实现，不是 native knowhere IVF 对等实现：
- 结构仅含 `dim/nlist/nprobe/centroids/lists/vectors`，无 `metric_type`、无统一 Index trait 接口、无序列化接口（`src/faiss/ivf.rs:4-11`, `13-23`, `25-160`）。
- 训练逻辑为“前 nlist 样本初始化 + 一次分配”，并未执行标准聚类训练流程（`src/faiss/ivf.rs:32-53`）。
- 搜索仅支持 L2 top-k 路径（`src/faiss/ivf.rs:81-102`），不覆盖 native IVF 家族的完整语义。

## Native 对比
native knowhere IVF 直接构建/调用 faiss IVF 索引，并使用完整 train/search/range/serialize 语义：
- 训练按 index type 构造 `faiss::IndexIVFFlat/IVFPQ/IVFScalarQuantizer` 并执行 `index->train(...)`（`/Users/ryan/Code/knowhere/src/index/ivf/ivf.cc:252-283`）。
- metric 由配置解析并传入索引构造（`ivf.cc:239-243`, `255-256`, `273`, `280`）。
- 同一实现提供 KNN + RangeSearch + 序列化（`ivf.cc:375-417`, `422-497`, `614-659`）。

## 影响
- Rust `ivf.rs` 与 native 的行为面、参数面和持久化面均不对齐，不能作为 parity 基线。
- 同名模块容易误导上层使用方，产生“看起来有 IVF，实际语义不兼容”的集成风险。

## 建议方向
- 将 `ivf.rs` 明确降级为 teaching/legacy 模块，或直接并入 `ivf_flat.rs` 的统一实现。
- parity 路径应只保留与 native 等价的 Index 接口实现（metric/nprobe/range/save-load 全覆盖）。
- 为 IVF 家族定义统一 compatibility checklist（参数、搜索语义、序列化互读）。

# ISSUE-HNSW-SERIALIZATION-COMPAT
**日期**: 2026-03-24 | **严重程度**: P1
## 问题
`src/faiss/hnsw.rs` 使用自定义二进制格式（magic=`HNSW`, version=3），按“向量数组 + ID 数组 + 每层邻接(id,dist)”写盘：
- header/config: `src/faiss/hnsw.rs:5866-5887`
- vectors/ids: `src/faiss/hnsw.rs:5891-5899`
- layer neighbors as `(nbr_id, dist)`: `src/faiss/hnsw.rs:5901-5914`
- load 严格要求 version=3: `src/faiss/hnsw.rs:5966-5975`

该格式不是 hnswlib 的 `saveIndex/loadIndex` 布局，且 layer-0 并非 hnswlib 的 linklist memory layout（`linklistsizeint + tableint[]`）。

## Native 对比
native knowhere 直接透传 hnswlib 序列化：
- `index_->saveIndex(writer)`（`/Users/ryan/Code/knowhere/src/index/hnsw/hnsw.cc:301-309`）
- `index_->loadIndex(reader)`（`/Users/ryan/Code/knowhere/src/index/hnsw/hnsw.cc:319-337`）

hnswlib 保存字段包含内部布局关键参数：`maxM_`, `maxM0_`, `mult_`, `ef_construction_`，以及 level0/linklists 原始内存块（`.../thirdparty/hnswlib/hnswlib/hnswalg.h:751-783`, `788-871`）。

## 影响
Rust HNSW 文件无法与 native knowhere/hnswlib 二进制互读。对生产中的跨语言迁移、离线构建在线加载、回归对标都会产生格式壁垒。

## 建议方向
定义兼容策略：
- 方案 A：新增 native-compatible codec（对齐 hnswlib binary）。
- 方案 B：保留当前 v3 格式，但明确标注“Rust-only”并提供转换工具。
- 对 Layer-0 邻接存储单独增加一致性验证用例（Rust 写 -> native 读 / native 写 -> Rust 读）。

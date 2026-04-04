# ISSUE-IVFPQ-METRIC-SERIALIZATION-PARITY
**日期**: 2026-03-24 | **严重程度**: P1
## 问题
`src/faiss/ivfpq.rs` 在 metric 语义和持久化格式上与 native IVF-PQ 存在关键偏差：
1. 搜索与 coarse assign 基本按 L2 写死，缺少按 `metric_type` 分发：
- coarse cluster distance 使用 `l2_distance_sq`（`src/faiss/ivfpq.rs:500-507`）
- centroid 分配 `find_nearest_centroid` 仅计算 L2（`src/faiss/ivfpq.rs:966-979`）
2. save/load 为自定义格式，并且 load 通过原始向量重新训练 PQ 再重建倒排，不是“读取已训练码本+codes”语义：
- 自定义 header `IVFPQ` + version（`src/faiss/ivfpq.rs:766-772`）
- load 后 `train_fine_quantizer(...)` + 重新编码（`src/faiss/ivfpq.rs:922-952`）

## Native 对比
native knowhere IVF-PQ 语义：
- metric 由配置解析后传入 `faiss::IndexIVFPQ`（`/Users/ryan/Code/knowhere/src/index/ivf/ivf.cc:239-243`, `268-274`）。
- 序列化走 FAISS reader/writer（二进制可互读），不做“加载时重新训练”行为：
  - Serialize: `faiss::write_index(...)`（`ivf.cc:614-623`）
  - Deserialize: `faiss::read_index(...)`（`ivf.cc:635-653`）
- 另外 native 在训练前会做 `MatchNlist/MatchNbits` 参数保护（`ivf.cc:193-221`, `270-272`），Rust 路径无对应自动修正。

## 影响
- IP/COSINE 场景下 IVF-PQ 行为可能与 native 明显偏离（recall 与距离排序不一致）。
- 加载成本被放大（需重训 PQ），且无法与 native 索引文件互读，破坏离线构建/在线加载链路。
- 配置鲁棒性不足：极端 nlist/nbits 可能触发训练质量或稳定性问题。

## 建议方向
- 为 IVF-PQ 全链路补齐 metric dispatch（coarse assign + ADC distance 方向一致）。
- 改为 native-compatible 持久化策略：保存/加载 PQ codebook 与倒排 codes，禁止 load 时重训。
- 引入 `MatchNlist/MatchNbits` 等价策略或显式参数校验，保证与 native 配置行为一致。

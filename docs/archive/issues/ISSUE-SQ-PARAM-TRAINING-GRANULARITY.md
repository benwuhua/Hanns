# ISSUE-SQ-PARAM-TRAINING-GRANULARITY
**日期**: 2026-03-24 | **严重程度**: P2
## 问题
`src/quantization/sq.rs` 当前 SQ 训练采用全局 `min/max`（跨所有维度一个 scale/offset），编码时统一 clip 到 `[0, 2^bit-1]`。  
这与 native knowhere 在 IVF-SQ 路径上直接复用 faiss `IndexIVFScalarQuantizer(QT_8bit)` 的训练/量化语义存在潜在偏差（通常更细粒度，不是单一全局范围）。

## Native 对比（引用文件+行号）
- Rust SQ 全局参数：
  - `src/quantization/sq.rs:52-57`（全局 min/max）
  - `src/quantization/sq.rs:59-65`（单一 scale/offset）
  - `src/quantization/sq.rs:73-76`（统一 clip+round）
- Native knowhere IVF-SQ8 走 faiss ScalarQuantizer：
  - `/Users/ryan/Code/knowhere/src/index/ivf/ivf.cc:276-282`
  - `/Users/ryan/Code/knowhere/src/index/ivf/ivf.cc:753-757`

## 影响
- 在高维、各维分布差异大的数据上，单一全局量化范围易放大量化误差。
- Rust SQ 与 native IVF_SQ8 的 recall/QPS 对标不稳定。

## 建议方向
- 增加“与 native 对齐模式”：按维（或分组）训练 scale/offset。
- 保留当前全局模式作为快速路径，但在配置中显式区分并默认走 parity 模式。
- 补充 parity benchmark：同数据集比较 Rust SQ vs native IVF_SQ8 的误差分布。

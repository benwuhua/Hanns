# ISSUE-HVQ-NATIVE-REGISTRY-GAP
**日期**: 2026-03-24 | **严重程度**: P2
## 问题
Rust 侧存在 `HvqQuantizer`（`src/quantization/hvq.rs`），包含随机旋转 + per-vector 头部参数（scale/offset）编码；  
但 native knowhere `src/index/` 当前无 HVQ 注册入口与对应索引类型，导致 HVQ 路径无法与 native 做同语义对标。

## Native 对比（引用文件+行号）
- Rust HVQ 关键行为：
  - `src/quantization/hvq.rs:20-31`（随机旋转初始化）
  - `src/quantization/hvq.rs:126-181`（编码 + 每向量附带 scale/offset）
  - `src/quantization/hvq.rs:233-240`（ADC 近似距离）
- Native knowhere IVF 仅注册 IVFPQ/IVFSQ8，无 HVQ：
  - `/Users/ryan/Code/knowhere/src/index/ivf/ivf.cc:748-757`

## 影响
- HVQ 功能目前更像 Rust 专有路径，不具备 native parity 可验证性。
- 若直接用于生产，跨实现一致性与回滚策略缺失。

## 建议方向
- 若目标是 native parity，先定义 HVQ 在 native 侧的等价类型与序列化协议。
- 在此之前将 HVQ 标记为 experimental，并建立单独 benchmark，不与 parity 指标混用。

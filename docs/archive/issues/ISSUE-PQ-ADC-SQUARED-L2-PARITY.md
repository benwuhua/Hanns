# ISSUE-PQ-ADC-SQUARED-L2-PARITY
**日期**: 2026-03-24 | **严重程度**: P1
## 问题
`src/faiss/pq.rs` 的 PQ 训练/编码/ADC 查表使用的是 `simd::l2_distance`，该函数返回的是开根号后的 L2，而不是 L2²。  
在 PQ/ADC 里按子空间累加时，通常应累加 squared L2（L2²）；把每个子空间先开根号再相加会改变排序语义，导致与 native IVF-PQ 距离定义不一致。

## Native 对比（引用文件+行号）
- Rust PQ 使用 `l2_distance`：
  - `src/faiss/pq.rs:215`（编码选最近中心）
  - `src/faiss/pq.rs:257`（build_distance_table）
- `l2_distance` 返回开根号：
  - `src/simd.rs:95-116`
  - `src/simd.rs:222-227`
- Native knowhere IVF-PQ 直接走 `faiss::IndexIVFPQ`（保持 faiss PQ 距离语义）：
  - `/Users/ryan/Code/knowhere/src/index/ivf/ivf.cc:268-274`
  - `/Users/ryan/Code/knowhere/src/index/ivf/ivf.cc:748-751`

## 影响
- PQ codebook 训练目标与搜索 ADC 度量和 native 不同，可能导致 recall/QPS 对齐困难。
- 同一参数（m/nbits/nprobe）下，Rust 与 native 输出分布不可直接对比。

## 建议方向
- PQ 训练、编码最近中心、ADC table 统一改为 `l2_distance_sq`（L2²）。
- 增加 parity test：同一随机种子下比较 Rust/native 的 code 分布与 top-k 重叠率。

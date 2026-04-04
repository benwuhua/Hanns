# ISSUE-PQ-NBITS-CONTRACT
**日期**: 2026-03-24 | **严重程度**: P1
## 问题
`src/faiss/pq.rs` 以 `k` 构造 PQ，并通过 `log2(k)` 推导 `nbits`，但未强制 `k` 为 2 的幂，也未校验 `k <= 256`。  
同时编码容器是 `u8`（每个子空间 1 byte），当 `k > 256` 时语义失配；当 `k` 非 2 的幂时，`nbits` 与真实码书大小不一致。

## Native 对比（引用文件+行号）
- Rust:
  - `src/faiss/pq.rs:23-26`（`nbits = log2(k)`）
  - `src/faiss/pq.rs:202`（每子空间一个 `u8`）
  - `src/faiss/pq.rs:222`（`best as u8`）
- Native knowhere 对 nbits 有显式 contract 与修正：
  - `/Users/ryan/Code/knowhere/src/index/ivf/ivf_config.h:53-57`（`nbits` 配置）
  - `/Users/ryan/Code/knowhere/src/index/ivf/ivf.cc:206-222`（`MatchNbits` 自动收敛到 1/2/4/8）
  - `/Users/ryan/Code/knowhere/src/index/ivf/ivf.cc:271-273`（构造 `faiss::IndexIVFPQ` 用 `nbits`）

## 影响
- 参数合法性与 native 不一致，导致训练行为不稳定或 silently degrade。
- 跨实现调参时出现“同名参数不同语义”。

## 建议方向
- 明确 PQ 参数 contract：优先采用 `(m, nbits)` 而非 `(m, k)`。
- 若保留 `k` 接口，增加校验：`k.is_power_of_two()` 且 `k <= 256`，否则报错。
- 提供与 native 一致的 `MatchNbits` 逻辑或等价约束。

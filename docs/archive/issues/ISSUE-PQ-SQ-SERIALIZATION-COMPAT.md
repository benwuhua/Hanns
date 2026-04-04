# ISSUE-PQ-SQ-SERIALIZATION-COMPAT
**日期**: 2026-03-24 | **严重程度**: P0
## 问题
当前 Rust 量化层（`src/faiss/pq.rs`, `src/quantization/sq.rs`）未提供与 native knowhere/faiss 索引文件的互读协议；  
而 native knowhere 的 IVF-PQ/IVF-SQ 走 faiss 标准序列化（`read_index`/`write_index` 对应格式）。这会导致 Rust 量化对象/索引难以与 native 索引文件直接互通。

## Native 对比（引用文件+行号）
- Native IVF 反序列化直接调用 faiss reader：
  - `/Users/ryan/Code/knowhere/src/index/ivf/ivf.cc:649-653`
  - `/Users/ryan/Code/knowhere/src/index/ivf/ivf.cc:671-675`
- Native IVF-PQ / IVF-SQ8 注册到统一体系：
  - `/Users/ryan/Code/knowhere/src/index/ivf/ivf.cc:748-757`
- Rust PQ/SQ 仅有内存 encode/decode/表构建，无 native-format 读写入口：
  - `src/faiss/pq.rs:201-279`
  - `src/quantization/sq.rs:68-280`

## 影响
- 无法做到“Rust 构建 -> native 加载”或“native 构建 -> Rust 加载”的无损互通。
- 线上迁移、灰度回滚、跨语言联调成本高，阻碍真正 parity 验证。

## 建议方向
- 增加 native-compatible serialization 层（至少 IVF-PQ/IVF-SQ8 索引级别）。
- 先实现离线转换器，再逐步内建 read/write 支持。
- 把“可互读”作为 parity 的验收门槛之一。

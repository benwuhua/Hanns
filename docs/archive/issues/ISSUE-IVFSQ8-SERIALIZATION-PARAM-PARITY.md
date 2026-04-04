# ISSUE-IVFSQ8-SERIALIZATION-PARAM-PARITY
**日期**: 2026-03-24 | **严重程度**: P2
## 问题
`src/faiss/ivf_sq8.rs` 的持久化格式与参数恢复策略与 native 不对齐：
- 使用自定义 magic `IVFSQ8` 二进制格式（`src/faiss/ivf_sq8.rs:721`, `787-792`），非 FAISS/knowhere 兼容格式。
- 反序列化时将 `nprobe` 固定回 8（`src/faiss/ivf_sq8.rs:879`, `883`），而不是恢复真实搜索参数。

## Native 对比
native knowhere IVF-SQ8 通过 FAISS reader/writer 做序列化互读：
- Serialize: `faiss::write_index(...)`（`/Users/ryan/Code/knowhere/src/index/ivf/ivf.cc:614-623`）
- Deserialize: `faiss::read_index(...)`（`ivf.cc:635-653`）

查询时 `nprobe` 来自配置并在 search 调用中显式传入（`ivf.cc:376`, `390`, `398-403`），不是固定常量。

## 影响
- Rust IVF-SQ8 文件不能直接与 native 互读，跨语言/跨组件部署链路受阻。
- load 后 `nprobe` 参数漂移（被重置为 8），会导致 recall/QPS 与预期配置不一致。

## 建议方向
- 改为与 native 对齐的 IVF-SQ8 codec（FAISS index binary 兼容或提供官方转换层）。
- 持久化并恢复完整搜索参数（至少 `nprobe`），避免加载后行为漂移。
- 增加 save/load 后参数等价性测试（含 `nprobe`、metric、nlist）。

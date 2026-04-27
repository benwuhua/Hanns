# HNSW-PQ scoped build+search verdict — 2026-04-25T11:22Z

Status: `build_and_search_win_at_higher_recall`。

结论：在官方 `benchmark_float.TEST_HNSW_PQ` FP32 emitted-row 范围内，Hanns HNSW-PQ 当前取得 scoped build+search milestone：

- Official FP32 refine_k=16：R@100=0.9576，QPS≈5050.505，build=84.542s
- Hanns refine_k=2：R@100=0.9756，QPS=9368.841，build=78.733s

限制：官方 full runner 仍在 FP32 行后 FP16 `bad optional access` 失败；Hanns graph build 日志显示 64 build threads，而官方日志显示 global build thread pool size 8。因此这是 HNSW-PQ FP32 emitted-row scoped milestone，不是 clean full-runner verdict，也不是“全面超越官方”。

Evidence:
- Hanns artifact: `docs/parity/raw/hanns-knowhere-hnsw-pq-aligned-1777115931.json`
- Hanns log/status: `docs/parity/raw/remote_background_20260425T111733Z_43762.log/status`
- Official log/status: `docs/parity/raw/remote_background_20260425T061341Z_98575.log/status`
- Official source provenance: `docs/parity/raw/remote_background_20260425T091234Z_11524.log/status`

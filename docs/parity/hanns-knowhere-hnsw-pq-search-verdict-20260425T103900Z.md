# HNSW-PQ scoped search verdict — 2026-04-25T10:39Z

Status: `search_win_at_near_official_recall`。

结论：Hanns HNSW-PQ 现在已经解除 raw-refine 可比性 blocker，并在官方 FP32 emitted row 范围内取得**搜索吞吐 scoped win**：Hanns refine_k=1 R@100=0.9546，接近官方 refine_k=16 R@100=0.9576（容差 0.005），QPS=17824.135，高于官方约 5050.505。

限制：Hanns build_s=89.475 仍慢于官方 build_s=84.542；官方 runner 仍是 FP32 行后 FP16 `bad optional access` 失败。因此这不是 build-time win，也不是 full-runner win，更不是“全面超越官方”。

Evidence:
- Hanns artifact: `docs/parity/raw/hanns-knowhere-hnsw-pq-aligned-1777113393.json`
- Hanns log/status: `docs/parity/raw/remote_background_20260425T103515Z_60789.log/status`
- Official log/status: `docs/parity/raw/remote_background_20260425T061341Z_98575.log/status`
- Official source provenance: `docs/parity/raw/remote_background_20260425T091234Z_11524.log/status`

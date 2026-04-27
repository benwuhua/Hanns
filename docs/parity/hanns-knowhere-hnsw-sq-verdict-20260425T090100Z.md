# Hanns vs official Knowhere HNSW-SQ scoped verdict — 2026-04-25 09:01Z

Status: **scoped win against the official emitted FP32 row**.

## Scope

Official `benchmark_float.TEST_HNSW_SQ` emits the FP32 row below, then the same test fails during later FP16/BF16 refine combinations with `bad optional access`. This verdict is therefore scoped to that archived FP32 row and is not a clean full-runner pass.

## Comparison

| side | config | R@100 | QPS/VPS | build_s |
| --- | --- | ---: | ---: | ---: |
| official zilliz Knowhere | HNSW_SQ FP32 M=16 efc=200 ef=128 refine_k=1 | 0.9531 | 30959.752 | 66.281 |
| Hanns | HNSW-SQ SQ8Refine M=16 efc=128 ef=256 | 0.9595 | 38561.913 | 59.486 |

Hanns is higher on recall, higher on search throughput, and lower on build time for this scoped comparison.

## Evidence

- Verdict JSON: `docs/parity/hanns-knowhere-hnsw-sq-verdict-20260425T090100Z.json`
- Hanns raw rows: `docs/parity/raw/hanns-knowhere-hnsw-sq-aligned-1777107479.json`
- Hanns log/status: `docs/parity/raw/remote_background_20260425T085756Z_79034.log`, `docs/parity/raw/remote_background_20260425T085756Z_79034.status`
- Official failed-run log/status with FP32 row: `docs/parity/raw/remote_background_20260425T061341Z_98575.log`, `docs/parity/raw/remote_background_20260425T061341Z_98575.status`
- Official source provenance: `docs/parity/raw/remote_background_20260425T091234Z_11524.log`, `docs/parity/raw/remote_background_20260425T091234Z_11524.status`

## Non-claims

- Not a clean full official `TEST_HNSW_SQ` pass.
- Not a HNSW-PQ verdict.
- Not an all-family / 全面领先 verdict.

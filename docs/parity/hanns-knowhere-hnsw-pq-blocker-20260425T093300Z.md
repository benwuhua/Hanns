# Hanns vs official Knowhere HNSW-PQ blocker — 2026-04-25 09:33Z

Status: **blocked / non-comparable for leadership**.

## Why blocked

Official `benchmark_float.TEST_HNSW_PQ` emits FP32 refined rows, then fails during later FP16 refine with `bad optional access`. The strongest emitted FP32 row is refine_k=16, R@100=0.9576, elapsed=1.980s (QPS 5050.505), build=84.542s.

Current Hanns `HnswPqIndex` stores lossy PQ codes and explicitly reports no raw data / unsupported `get_vector_by_ids`, so it cannot provide the FLAT refine surface used by official HNSW_PQ rows. Therefore a HNSW-PQ leadership claim is blocked until Hanns implements an aligned raw-vector refine path or an accepted equivalent.

## Evidence

- Verdict JSON: `docs/parity/hanns-knowhere-hnsw-pq-blocker-20260425T093300Z.json`
- Official failed-run FP32 rows: `docs/parity/raw/remote_background_20260425T061341Z_98575.log/status`
- Official qps-runner absence: `docs/parity/raw/official_qps_gtest_list_20260425T065500Z.log`
- Official source provenance: `docs/parity/raw/remote_background_20260425T091234Z_11524.log/status`
- Hanns capability proof: `docs/parity/raw/remote_background_20260425T093142Z_84447.log/status`

## Non-claims

- Not a Hanns HNSW-PQ win.
- Not a full-family / 全面领先 verdict.

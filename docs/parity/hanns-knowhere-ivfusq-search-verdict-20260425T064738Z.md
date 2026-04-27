# Hanns vs official Knowhere IVF-USQ/RabitQ search verdict (HannsDB-x86)

Status: **scoped search-throughput win**
Scope: IVF-USQ/RabitQ on SIFT1M, `top_k=100`, benchmark_float-style one-vector-per-query throughput. This is **not** a build-time win and it is not a full-family/all-family claim.

## Comparison row

| Side | Runner | Params | Recall@100 | Throughput | Build(s) |
|---|---|---|---:|---:|---:|
| Official Zilliz Knowhere IVF_RABITQ(FP32) | benchmark_float | nlist=1024, nprobe=64 | 0.6146 | 7727.975 | not claimed from cached rerun |
| Hanns IVF-USQ | aligned Rust lane | bits=8, nlist=1024, nprobe=4 | 0.6144 | 19661.756 | 85.762 |

Hanns is within the configured recall tolerance (`0.001`) and is `2.54x` faster on the search-throughput metric for this scoped lane.

## Evidence

- Verdict JSON: `docs/parity/hanns-knowhere-ivfusq-search-verdict-20260425T064738Z.json`
- Hanns aligned JSON: `docs/parity/raw/hanns-knowhere-ivfusq-aligned-1777099565.json`
- Hanns remote log/status: `docs/parity/raw/remote_background_20260425T064447Z_78700.log`, `docs/parity/raw/remote_background_20260425T064447Z_78700.status`
- Official remote log/status: `docs/parity/raw/remote_background_20260425T062718Z_21446.log`, `docs/parity/raw/remote_background_20260425T062718Z_21446.status`
- Official qps gtest list proving no IVF_RABITQ qps test on commit `868634bc5230a71a84371fea5f5b674b805643ea`: `docs/parity/raw/official_qps_gtest_list_20260425T065500Z.log`
- Validator: `python3 scripts/hanns_knowhere_benchmark.py validate-ivfusq-verdict docs/parity/hanns-knowhere-ivfusq-search-verdict-20260425T064738Z.json`

## Remaining boundary

Official `benchmark_float_qps` does not expose IVF_RABITQ on this commit, so this verdict uses official `benchmark_float.TEST_IVF_RABITQ`, not qps-runner evidence. HNSW-SQ/HNSW-PQ official `benchmark_float` probes currently fail with `bad optional access` and remain separate blockers.

# Hanns vs Zilliz Knowhere HNSW near-0.80 verdict (2026-04-24)

Status: **win** for the official HNSW near-0.80 lane on **HannsDB-x86**.

## Contract

- Dataset: SIFT1M (`sift-128-euclidean`), `top_k=100`, 10,000 queries.
- Metric of record: `Recall@100` plus one-vector-per-query 8-thread QPS/VPS.
- Build time is required and compared under an 8-build-thread contract.
- Official Knowhere source: Zilliz `zilliztech/knowhere`, commit `868634bc5230a71a84371fea5f5b674b805643ea`.

## Winning row

| Impl | Variant / Params | Recall@100 | 8T QPS/VPS | build_s |
|---|---|---:|---:|---:|
| Official Zilliz Knowhere | HNSW(FP16), M=16, efConstruction=100, ef=100 | 0.9178 | 30013.983 | 29.339 |
| Hanns | HNSW, M=6, efConstruction=34, ef=850 | 0.9198 | 30972.456 | 23.878 |

Margins: Recall@100 +0.0020, QPS/VPS +958.473, build_s -5.461s.

## Evidence

- Verdict JSON: `docs/parity/hanns-knowhere-hnsw-verdict-20260424T162100Z.json`
- Hanns artifact: `docs/parity/hanns-knowhere-hnsw-aligned-1777047450.json`
- Hanns remote log: `docs/parity/raw/remote_background_20260424T161705Z_52013.log`
- Official remote log: `docs/parity/raw/native_benchmark_20260424T155636Z_41648.log`

## Scope note

This is not a claim that every HNSW recall lane or every index family is done. It closes the HNSW near-0.80 lane; near-0.95 and the remaining families stay as separate targets.

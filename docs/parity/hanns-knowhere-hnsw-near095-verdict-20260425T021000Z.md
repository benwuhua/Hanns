# Hanns vs Zilliz Knowhere HNSW near-0.95 verdict (2026-04-25)

Status: **win** for the official HNSW near-0.95 lane on **HannsDB-x86**.

## Contract

- Dataset: SIFT1M (`sift-128-euclidean`), `top_k=100`, 10,000 queries.
- Metric of record: `Recall@100` plus one-vector-per-query 8-thread QPS/VPS.
- Build time is required and compared under an 8-build-thread contract.
- Official Knowhere source: Zilliz `zilliztech/knowhere`, commit `868634bc5230a71a84371fea5f5b674b805643ea`.

## Winning row

| Impl | Variant / Params | Recall@100 | 8T QPS/VPS | build_s |
|---|---|---:|---:|---:|
| Official Zilliz Knowhere | HNSW(FP16), M=16, efConstruction=100, ef=138 | 0.9500 | 23370.333 | 29.339 |
| Hanns | HNSW, M=8, efConstruction=34, ef=880 | 0.9506 | 27220.125 | 28.884 |

Margins: Recall@100 +0.0006, QPS/VPS +3849.792, build_s -0.455s.

## Evidence

- Verdict JSON: `docs/parity/hanns-knowhere-hnsw-near095-verdict-20260425T021000Z.json`
- Hanns artifact: `docs/parity/hanns-knowhere-hnsw-aligned-1777082864.json`
- Hanns remote log: `docs/parity/raw/remote_background_20260425T020623Z_61280.log`
- Official remote log: `docs/parity/raw/native_benchmark_20260424T155636Z_41648.log`

## Scope note

This closes the HNSW near-0.95 lane only. It does not claim every remaining family is complete.
